import torch
import numpy as np
from collections import defaultdict
import pickle, requests
from requests.adapters import HTTPAdapter, Retry
from train.llm_server  import GeneralLLMServer
import re
import json
from PIL import Image
from typing import Optional, Dict, Any, Callable
from train.llm_server import LLMRequest
from io import BytesIO


def geneval_score(url):
    """Submits images to GenEval and computes a reward.
    """
    batch_size = 64
    sess = requests.Session()
    retries = Retry(
        total=3, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict=False, return_reason=False):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        all_reasons = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)
            if return_reason:
                all_reasons.append(response_data["reasons"])
            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)
        if return_reason:
            return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict, all_reasons
        else:
            return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn



GENEVAL_TRAIN_SYSTEM_PROMPT = """You are an expert image evaluator.

Your task is to determine whether the given image faithfully satisfies the visual instruction and the expectation checklist.

Follow these rules strictly:
1. The image must match **all** expectations, including:
   - Object classes
   - Counts of each object
   - Colors of each object
   - Spatial position within the image (e.g., "above", "below", based on real pixel position)
   - Size and relative scale of objects
2. The image must appear as a **natural, coherent, photo-like single image**.
   - Do NOT allow stylized images (e.g., cartoons, sketches, anime).
   - Do NOT allow collage-style or multi-panel images. Only one consistent, realistic scene is acceptable.
3. Be very strict and conservative in your judgment.

Return your result as a JSON object using this format:
{
  "correct": 1 if the image fully satisfies all expectations, else 0,
}
"""
#   "reason": "You may explain in detail what is missing or incorrect"
def metadata_to_explanation(metadata: dict) -> str:
    parts = []

    def format_item(item: dict) -> str:
        obj = item["class"]
        count = item.get("count", 1)
        color = item.get("color", None)
        region = item.get("region", None)
        size = item.get("size", None)
        noun = f"{count} {obj + 's' if count > 1 else obj}"
        desc_parts = []
        if color:
            desc_parts.append(f"{color}-colored")
        if size:
            desc_parts.append(size)
        if desc_parts:
            noun = f"{' '.join(desc_parts)} {noun}"
        if region:
            return f"{noun} located in the {region} part of the image"
        else:
            return f"{noun} present in the image"

    for item in metadata.get("include", []):
        parts.append(f"- {format_item(item)}.")
    for item in metadata.get("exclude", []):
        obj = item["class"]
        count = item.get("count", 1)
        noun = f"{obj + 's' if count > 1 else obj}"
        parts.append(f"- No more than {count - 1} {noun} should appear.")

    return "This image should contain:\n" + "\n".join(parts)


def extract_json_from_response(text: str):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))
    else:
        return {"correct": 0}
    

def geneval_plus_train_reward(url: str, 
    model_name: str = "Qwen2.5-VL-72B-Instruct-AWQ",
    api_key: str = "EMPTY", post_processor: Optional[Callable[[str], Any]] = extract_json_from_response,
    client_type: str = "openai",
    system_prompt: str = GENEVAL_TRAIN_SYSTEM_PROMPT):
    server = GeneralLLMServer(
        url=url,
        model_name=model_name,
        api_key=api_key,
        post_processor=post_processor,
        client_type=client_type,
        max_retries=3,
        retry_delay=0.5,
    )

    def _reward_fn(image: Image.Image, prompt: str, metadata: dict):
        instruction = prompt.strip()
        explanation = metadata_to_explanation(metadata)
        user_input = f"Instruction: {instruction}\n\nExplanation: {explanation}"
        request = LLMRequest(
            content=[user_input, image],
            system_prompt=system_prompt
        )
        response = server.send_request(request)
        response = response['processed_response']
        if response is None:
            print(f"[GENEVAL_PLUS_REWARD_FN] [Error] Response is None")
            return (0, "")
        reward = response.get('correct', 0)
        reason = response.get('reason', "")
        return (reward, reason)
    
    return _reward_fn



YN_SYSTEM_PROMPT_QWEN = '''
You are tasked with conducting a careful examination of the provided image. Based on the content of the image, please answer the questions according to the instructions:

Questions:
##YNQuestions##

Instructions:
1. Each question is numbered and formatted as "Q1:", "Q2:", etc.
2. Answer each question with either yes or no.
3. Return your answers in JSON format only.
4. The JSON should contain the question number as the key and the answer (yes or no) as the value.
5. The order of answers must correspond exactly to the order of the questions.
6. Do not include any explanations, reasoning, or additional content - only the JSON output.
7. Ensure the number of answers equals the number of questions.

Output format example:
{
  "Q1": "yes",
  "Q2": "no",
  "Q3": "yes",
}
'''

def vqa_reward_fn(url: str, 
    model_name: str = "Qwen2.5-VL-72B-Instruct-AWQ",
    api_key: str = "EMPTY", 
    post_processor: Optional[Callable[[str], Any]] = extract_json_from_response,
    client_type: str = "openai",
    return_number: bool = False):
    server = GeneralLLMServer(
        url=url,
        model_name=model_name,
        api_key=api_key,
        post_processor=post_processor,
        max_retries=3,
        retry_delay=0.5,
        client_type=client_type,
    )


    def _reward_fn(image: Image.Image, prompt: str, metadata: dict):
        question_list = metadata.get("yn_question_list", [])
        total_questions = len(question_list)
        question_str = "\n".join([f"Q{i+1}: {question}" for i, question in enumerate(question_list)])
        gt_answers = metadata.get("yn_answer_list", [])
        user_input = YN_SYSTEM_PROMPT_QWEN.replace("##YNQuestions##", question_str)
        request = LLMRequest(
            content=[user_input, image],
        )
        response = server.send_request(request)['processed_response']
        reward = 0
        try:
            correct_count = 0
            if len(response.values()) != len(question_list):
                print(f"[YN_REWARD_FN] [Warning] The number of answers does not match the number of questions: {len(response.values())} != {len(question_list)}")
            for pred, gt in zip(response.values(), gt_answers):
                if pred.lower().strip() == gt.lower().strip():
                    reward += 1
                    correct_count += 1
            reward = float(reward / len(response.values()))
            reward = round(reward, 2)
            if return_number:
                return reward, (correct_count, total_questions)
            return reward
        except Exception as e:
            reward = 0.0
        return reward
    
    return _reward_fn



TIIF_EVAL_SYSTEM_PROMPT = '''
You are tasked with conducting a careful examination of the provided image. Based on the content of the image, please answer the following yes or no questions:

Questions:
##YNQuestions##

Note that:
1. Each answer should be on a separate line, starting with "yes" or "no", followed by the reason.
2. The order of answers must correspond exactly to the order of the questions.
3. Each question must have only one answer.
4. Directly return the answers to each question, without any additional content.
5. Each answer must be on its own line!
6. Make sure the number of output answers equal to the number of questions!
'''

def format_questions_prompt(raw_prompt, questions):
    question_texts = [item.strip() for item in questions]
    formatted_questions = "\n".join(question_texts)
    formatted_prompt = raw_prompt.replace("##YNQuestions##", formatted_questions)
    return formatted_prompt

def extract_yes_no(model_output, questions):
    lines = [line.strip() for line in model_output.strip().split('\n') if line.strip()]
    preds = []
    for idx, line in enumerate(lines):
        m = re.match(r'^(yes|no)\b', line.strip(), flags=re.IGNORECASE)
        if m:
            preds.append(m.group(1).lower())
        else:
            continue
    if len(preds) != len(questions):
        raise ValueError(f"Preds count {len(preds)} != questions count {len(questions)}")
    return preds


def tiif_reward_fn(url: str, 
    model_name: str = "Qwen2.5-VL-72B-Instruct-AWQ",
    api_key: str = "EMPTY", 
    post_processor: Optional[Callable[[str], Any]] = None):
    server = GeneralLLMServer(
        url=url,
        model_name=model_name,
        api_key=api_key,
        post_processor=post_processor,
    )

    def _reward_fn(image: Image.Image, prompt: str, metadata: dict,):
        question_list = metadata.get("yn_question_list", [])
        gt_answers = metadata.get("yn_answer_list", [])
        prompt = format_questions_prompt(TIIF_EVAL_SYSTEM_PROMPT, question_list)
        request = LLMRequest(
            content=[prompt, image],
        )
        response = server.send_request(request)['response']
        reward = 0
        try:
            preds = extract_yes_no(response, question_list)
            # count the number of "yes" in preds
            for pred, gt in zip(preds, gt_answers):
                if pred == gt:
                    reward += 1
            reward = float(reward / len(preds))
            return reward
        except Exception as e:
            print(f"[TIIF_RAW_REWARD_FN] [Error] {e}")
            reward = 0.0
            return reward
    
    return _reward_fn


GENEVAL_PLUS_RAW_SYSTEM_PROMPT = """You are an expert image evaluator.

Your task is to determine whether the given image faithfully satisfies the visual instruction and the expectation checklist.

Follow these rules strictly:
1. The image must match **all** expectations, including:
   - Object classes
   - Counts of each object
   - Colors of each object
   - Spatial position within the image (e.g., "above", "below", based on real pixel position)
   - Size and relative scale of objects
2. The image must appear as a **natural, coherent, photo-like single image**.
   - Do NOT allow stylized images (e.g., cartoons, sketches, anime).
   - Do NOT allow collage-style or multi-panel images. Only one consistent, realistic scene is acceptable.
3. Be very strict and conservative in your judgment. 

Return your result as a JSON object using this format:
{
  "correct": 1 if the image fully satisfies all expectations, else 0,
  "reason": "You may explain in detail what is missing or incorrect"
}
"""

def geneval_plus_eval_reward(url: str, 
    model_name: str = "Qwen2.5-VL-72B-Instruct-AWQ",
    api_key: str = "EMPTY", 
    post_processor: Optional[Callable[[str], Any]] = extract_json_from_response):
    server = GeneralLLMServer(
        url=url,
        model_name=model_name,
        api_key=api_key,
        max_retries=3, 
        retry_delay=0.5,
        post_processor=post_processor,
    )   

    def _reward_fn(image: Image.Image, prompt: str, metadata: dict, return_reason: bool = False):
        instruction = prompt.strip()
        explanation = metadata_to_explanation(metadata)
        user_input = f"Instruction:\n{instruction}\n\nExplanation checklist:\n{explanation}"

        request = LLMRequest(
            content=[user_input, image],
            system_prompt=GENEVAL_PLUS_RAW_SYSTEM_PROMPT
        )
        response = server.send_request(request)['processed_response']
        try:
            reward = float(response.get('correct', 0))
            reason = response.get('reason', "")
        except Exception as e:
            reward = 0.0
            reason = ""
        return reward, reason
    return _reward_fn