# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os, sys, json
from copy import deepcopy
from dataclasses import dataclass, field
import time
from concurrent import futures
from datetime import timedelta  
import torch
from typing import List
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.qwen2.modeling_qwen2 import Qwen2RMSNorm
from torch import nn
from train.train_utils import (
    create_logger, 
    get_latest_ckpt,
    save_result_as_html,
    compute_advantages_over_policy_groups,
    compute_advantages_for_multi_round_rollout,
    EvalStats,
)
from train.fsdp_utils import (
    FSDPCheckpoint,
)
from train.fsdp2_utils import (
    FSDP2Checkpoint,
    FSDP2Config,
    apply_fsdp2_with_activation_checkpointing,
    clip_grad_norm_fsdp2,
    fsdp2_ema_update,
    fsdp2_lazy_init_root,
)
from train.rollout_controller import MultiRoundRolloutController,RolloutStepResult
from train.data_utils import (
    GenevalPromptDataset,
    DistributedKRepeatSampler,
    TIIFDataset,
)
from data.transforms import ImageTransform
from modeling.diffusion.sde_sampler import SDESampler
from train.reward_func import geneval_plus_train_reward, vqa_reward_fn, tiif_reward_fn, geneval_plus_eval_reward, geneval_score
import wandb
from tqdm import tqdm


@dataclass
class ModelArguments:
    llm_path: str = field(default="hf/Qwen2.5-0.5B-Instruct/")
    llm_qk_norm: bool = field(default=True)
    tie_word_embeddings: bool = field(default=False)
    layer_module: str = field(default="Qwen2MoTDecoderLayer")
    vae_path: str = field(default="flux/vae/ae.safetensors")
    vit_path: str = field(default="hf/siglip-so400m-14-980-flash-attn2-navit/")
    max_latent_size: int = field(default=64)
    latent_patch_size: int = field(default=2)
    vit_patch_size: int = field(default=14)
    vit_max_num_patch_per_side: int = field(default=70)
    connector_act: str = field(default="gelu_pytorch_tanh")
    interpolate_pos: bool = field(default=False)
    vit_select_layer: int = field(default=-2)
    vit_rope: bool = field(default=False)

    text_cond_dropout_prob: float = field(default=0.1)
    vae_cond_dropout_prob: float = field(default=0.3)
    vit_cond_dropout_prob: float = field(default=0.3)


@dataclass
class DataArguments:
    dataset_config_file: str = field(default="data/configs/online_rl_example.yaml")
    prefetch_factor: int = field(default=2)
    num_workers: int = field(default=4)
    max_num_tokens_per_sample: int = field(default=16384)
    max_num_tokens: int = field(default=36864)
    prefer_buffer_before: int = field(default=16384)
    max_buffer_size: int = field(default=50)
    data_seed: int = field(default=42)
    online_batch_size: int = field(default=1)  # 在线推理的批次大小

    policy_group_size: int = field(default=1)

    vae_max_image_size: int = field(default=512)
    vae_min_image_size: int = field(default=256)
    vae_image_stride: int = field(default=16)
    vit_max_image_size: int = field(default=504)
    vit_min_image_size: int = field(default=252)
    vit_image_stride: int = field(default=14)
    data_path: str = field(default="prompts.txt")
    dataset_name: str = field(default="hps")
    train_data_path: str = field(default="prompts_train.txt")
    val_data_path: str = field(default="prompts_val.txt")


@dataclass
class TrainingArguments:
    debug: bool = field(default=False)
    exp_name: str = field(default="")

    log_dir: str = field(default="your_exps/")
    visual_gen: bool = field(default=True)
    visual_und: bool = field(default=True)

    mydir: str = field(default="")
    results_dir: str = field(default="results")
    checkpoint_dir: str = field(default="results/checkpoints")
    wandb_project: str = field(default="interleave_thinking")
    wandb_name: str = field(default="run")
    wandb_runid: str = field(default="trial")
    wandb_resume: str = field(default="allow")
    wandb_offline: bool = field(default=False)
    global_seed: int = field(default=4396)
    auto_resume: bool = field(default=False)
    resume_from: str = field(default=None)
    resume_model_only: bool = field(default=False)
    finetune_from_ema: bool = field(default=False)
    log_every: int = field(default=5)
    save_every: int = field(default=50)
    total_steps: int = field(default=2000)

    warmup_steps: int = field(default=50)
    lr_scheduler: str = field(default="constant")
    lr: float = field(default=5e-6)
    min_lr: float = field(default=1e-7)
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.95)
    weight_decay: float = field(default=0)
    eps: float = field(default=1e-15)
    ema: float = field(default=0)
    max_grad_norm: float = field(default=1.0)
    mse_weight: float = field(default=1.0)
    ce_weight: float = field(default=1.0)
    ce_loss_reweighting: bool = field(default=False)
    expected_num_tokens: int = field(default=32768)

    num_replicate: int = field(default=1)
    num_shard: int = field(default=8)
    sharding_strategy: str = field(default="SHARD_GRAD_OP")
    backward_prefetch: str = field(default="BACKWARD_PRE")
    cpu_offload: bool = field(default=False)

    freeze_llm: bool = field(default=False)
    freeze_vit: bool = field(default=True)
    freeze_vae: bool = field(default=True)
    freeze_und: bool = field(default=False)
    freeze_text_transformer: bool = field(default=False)
    freeze_diffusion_transformer: bool = field(default=False)
    copy_init_moe: bool = field(default=True)
    use_flex: bool = field(default=False)
    eval_only: bool = field(default=False)

    # RL specific parameters
    think_mode: bool = field(default=True)
    tune_text_cot: bool = field(default=True)
    tune_image_cot: bool = field(default=True)
    rounds: int = field(default=2) # multi-round rollout
    group_size: int = field(default=16)  # [G]RPO
    kl_weight_text: float = field(default=0.0)  # KL penalty for text
    kl_weight_image: float = field(default=0.0)  # KL penalty for image
    rollout_vis_step: int = field(default=20)  # 每多少步可视化推理结果
    eval_freq: int = field(default=50)  # 每多少步评估一次
    reward_fn: str = field(default="geneval")  # 奖励函数
    format_reward_weight: float = field(default=1.0) # 格式化奖励的权重
    pack_start_round_idx: int = field(default=0) # 从第几轮开始打包rollout结果为train input
    enable_sde_gen: bool = field(default=True)
    enable_sde_edit: bool = field(default=True)
    grpo_clip_ratio: float = field(default=0.2)

    # reward server specific parameters
    reward_server_urls: str = field(
        default=""
    )  # if have multiple ips, use comma to seperate
    reward_server_port: str = field(default="")
    reward_model_name: str = field(default="Qwen2.5-VL-72B-Instruct-AWQ")
    reward_api_key: str = field(default="EMPTY")
    client_type: str = field(default="openai")
    # inference
    eval_rounds: int = field(default=4)
    rollout_on_same_noise: bool = field(default=True)
    timestep_sample_ratio: float = field(default=0.1)
    max_output_token_n_gen: int = field(default=196)
    max_output_token_n_edit: int = field(default=384)
    do_sample: bool = field(default=True)
    text_temperature: float = field(default=0.8)
    top_k: int = field(default=-1)
    top_p: float = field(default=0.9)
    use_static_kv_cache: bool = field(default=True)
    cfg_text_scale: float = field(default=4.0)
    cfg_img_scale: float = field(default=1.5)
    cfg_interval: tuple = field(default=(0.4, 1.0))
    cfg_interval_low: float = field(default=0.4)
    cfg_interval_high: float = field(default=1.0)
    timestep_shift: float = field(default=3.0)
    num_timesteps_gen: int = field(default=15)  # should be fixed
    num_timesteps_edit: int = field(default=10)  # should be fixed
    cfg_renorm_min: float = field(default=0.0)
    cfg_renorm_type_gen: str = field(default="global")
    cfg_renorm_type_edit: str = field(default="text_channe")
    image_size: int = field(default=512)
    eta_mode: str = field(default='monotonic')
    constant_eta_gen: float = field(default=1.0)
    constant_eta_edit: float = field(default=1.0)
    max_sde_timestep_idx_for_edit: int = field(default=2)
    resume_step: int = field(default=-1)


def build_debug_reward_fns(reward_fn_name: str):
    if reward_fn_name == "geneval":
        def _geneval_reward(images, prompts, metadatas, only_strict=False, return_reason=False):
            del prompts, metadatas, only_strict, return_reason
            batch_size = len(images)
            zeros = [0.0] * batch_size
            return zeros, zeros, zeros, {}, {}

        return _geneval_reward, _geneval_reward

    if reward_fn_name == "geneval_plus":
        def _geneval_plus_reward(image, prompt, metadata, return_reason: bool = False):
            del image, prompt, metadata, return_reason
            return 0.0, ""

        return _geneval_plus_reward, _geneval_plus_reward

    if reward_fn_name == "tiif":
        def _tiif_reward(image, prompt, metadata, return_reason: bool = False):
            del image, prompt, metadata, return_reason
            return 0.0

        return _tiif_reward, _tiif_reward

    raise ValueError(f"Invalid reward function for debug mode: {reward_fn_name}")


def gather_text_policy_logps(
    logits: torch.Tensor,
    label_ids: torch.LongTensor,
    *,
    do_sample: bool,
    temperature: float,
    topk: int,
    top_p: float,
) -> torch.Tensor:
    del do_sample, temperature, topk, top_p
    return (
        logits.log_softmax(dim=-1)
        .gather(dim=-1, index=label_ids.unsqueeze(-1))
        .squeeze(-1)
    )


DATASET_EVAL_KEYS = {
    "geneval": [
        "single_object", "two_object", "counting", "colors", "position", "color_attr",
    ],
    "geneval_plus": [
        "color_attr", "spatial_count_attr", "color_spatial_attr", "color_count_attr",
        "multi_object_count_attr", "size_spatial_attr", "counting",
    ],
    "tiif": [
        "numeracy+2d", "comparison", "action+texture", "comparison+3d", "texture+color",
        "numeracy+3d", "numeracy", "action+2d", "color+3d", "color+texture",
        "differentiation+3d", "comparison+2d", "shape+2d", "negation+3d", "comparison+texture",
        "3d_spatial_relation", "shape+3d", "negation+2d", "differentiation", "negation+color",
        "action+color", "text", "texture+2d", "differentiation+color", "real_world",
        "differentiation+2d", "action+3d", "2d_spatial_relation", "numeracy+texture", "color+2d",
        "differentiation+texture", "negation", "negation+texture", "comparison+color",
        "shape+texture", "style", "shape+color", "texture+3d", "numeracy+color",
    ],
}

METADATA_TAG_KEY = {
    "geneval": "tag",
    "geneval_plus": "tag",
    "tiif": "type",
}

_REFLECTION_TEMPLATE = (
    "The description of the target image is: {prompt}\n"
    "How to further edit the provided image to make it consistent with the target description? "
    "Please provide concrete editing instructions in a single sentence. "
    "If no editing operation is needed, answer: no further edit needed."
)


def _extract_reward_value(reward_result, dataset_name: str):
    """Extract scalar reward value from the async reward future result."""
    if dataset_name == "geneval":
        _scores, rewards, _strict, _group_dict, _group_strict_dict = reward_result
        return rewards[0]
    elif dataset_name == "geneval_plus":
        return reward_result[0]
    elif dataset_name == "tiif":
        return reward_result
    raise ValueError(f"Unknown dataset: {dataset_name}")


def evaluate_multi_round(
    val_dataset,
    rollout_controller: MultiRoundRolloutController,
    training_args: TrainingArguments,
    data_args: DataArguments,
    curr_step: int,
    generation_kwargs: dict = None,
    eval_batch_size: int = 16,
    eval_rounds: int = 4,
):
    dataset_name = data_args.dataset_name
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    validation_set_len = len(val_dataset)
    if validation_set_len == 0:
        return {}
    base_val_per_rank = validation_set_len // world_size
    remainder = validation_set_len % world_size
    start_idx = rank * base_val_per_rank + min(rank, remainder)
    end_idx = start_idx + base_val_per_rank + (1 if rank < remainder else 0)
    val_data = [val_dataset[i] for i in range(start_idx, end_idx)]
    val_prompts = [item['prompt'] for item in val_data]
    val_metadata = [item['metadata'] for item in val_data]

    reflection_prompts = [_REFLECTION_TEMPLATE.format(prompt=p) for p in val_prompts]
    dataset_keys = DATASET_EVAL_KEYS[dataset_name]
    tag_key = METADATA_TAG_KEY[dataset_name]

    round_eval_stats = [
        EvalStats(basic_info=[{"Round": r}], track_keys=dataset_keys)
        for r in range(eval_rounds)
    ]
    round_improve_stats = [
        EvalStats(basic_info=[{"Round": r}], track_keys=["improved", "worse", "no_change"])
        for r in range(eval_rounds)
    ]
    round_format_rewards = [[] for _ in range(eval_rounds)]
    round_mean_cot_length = [[] for _ in range(eval_rounds)]
    round_improve_rewards = [0.0 for _ in range(eval_rounds)]
    finished_cnt = 0
    finished_steps = 0
    finish_stats = {f"Round_{r}_finished_cnt": 0 for r in range(eval_rounds)}
    last_batch_results = None

    def _distributed_mean(values: list[float]) -> float:
        local_sum = torch.tensor(sum(values), dtype=torch.float, device="cuda")
        local_count = torch.tensor(len(values), dtype=torch.float, device="cuda")
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        if local_count.item() == 0:
            return 0.0
        return (local_sum / local_count).item()

    with rollout_controller.context():
        for i in range(0, len(val_prompts), eval_batch_size):
            g = torch.Generator(device="cuda")
            g.manual_seed((rank + 1) * (len(val_prompts) // eval_batch_size + 1) + i)
            num_image_tokens = (
                generation_kwargs["image_shapes"][0]
                * generation_kwargs["image_shapes"][1]
                // 16**2
            )
            generation_kwargs["initial_noise"] = [
                torch.randn(num_image_tokens, 64, dtype=torch.bfloat16, generator=g, device="cuda")
                for _ in range(eval_rounds)
            ]
            batch_prompts = val_prompts[i:i + eval_batch_size]
            batch_metadata = val_metadata[i:i + eval_batch_size]
            batch_reflection = reflection_prompts[i:i + eval_batch_size]
            batch_results = rollout_controller.rollout_for_eval(
                rounds=eval_rounds,
                prompts=batch_prompts,
                reflection_prompts=batch_reflection,
                prompt_metadata=batch_metadata,
                generator=g,
                **generation_kwargs,
            )
            last_batch_results = batch_results

            for round_idx in range(eval_rounds):
                round_mean_cot_length[round_idx].append(
                    batch_results[f"Round_{round_idx}/mean_cot_length"]
                )

            for sample_idx, per_sample_reward in enumerate(batch_results["per_sample_rewards"]):
                finished_cnt += int(batch_results["per_sample_finished"][sample_idx])
                finished_step = batch_results["per_sample_finished_steps"][sample_idx]
                finished_steps += int(finished_step)
                finish_stats[f"Round_{finished_step - 1}_finished_cnt"] += 1

                for round_idx, per_round_reward in enumerate(per_sample_reward):
                    reward_value = _extract_reward_value(per_round_reward.result(), dataset_name)
                    per_sample_reward[round_idx] = reward_value
                    tag = batch_metadata[sample_idx][tag_key]
                    round_eval_stats[round_idx].update(tag, reward_value)
                    round_format_rewards[round_idx].append(
                        batch_results["per_sample_format_rewards"][sample_idx][round_idx]
                    )
                    if round_idx > 0:
                        prev_reward = int(per_sample_reward[round_idx - 1])
                        curr_reward = int(per_sample_reward[round_idx])
                        delta = curr_reward - prev_reward
                        round_improve_rewards[round_idx] += delta / validation_set_len
                        if delta > 0:
                            round_improve_stats[round_idx].update("improved", 1 / validation_set_len)
                        elif delta < 0:
                            round_improve_stats[round_idx].update("worse", 1 / validation_set_len)
                        else:
                            round_improve_stats[round_idx].update("no_change", 1 / validation_set_len)

    eval_log = {}
    if last_batch_results is not None:
        save_inference_results(
            last_batch_results, curr_step, training_args.results_dir, mode="eval", num_samples_to_save=8
        )
    log_stats = [stat.aggregate() for stat in round_eval_stats]
    round_improve_stats = [
        stat.aggregate(normalize=False, log_overall=False) for stat in round_improve_stats
    ]
    finished_cnt = torch.tensor(finished_cnt, device="cuda", dtype=torch.float) / validation_set_len
    finished_steps = torch.tensor(finished_steps, device="cuda", dtype=torch.float) / validation_set_len
    dist.all_reduce(finished_cnt, op=dist.ReduceOp.SUM)
    dist.all_reduce(finished_steps, op=dist.ReduceOp.SUM)
    eval_log[f"Eval_{dataset_name}/Finished_rate"] = finished_cnt.item()
    eval_log[f"Eval_{dataset_name}/Finished_steps"] = finished_steps.item()
    for round_idx in range(eval_rounds):
        eval_log[f"Eval_{dataset_name}/Round_{round_idx}_format_reward_mean"] = _distributed_mean(
            round_format_rewards[round_idx]
        )
        eval_log[f"Eval_{dataset_name}/Round_{round_idx}_mean_cot_length"] = _distributed_mean(
            round_mean_cot_length[round_idx]
        )
        for key, value in log_stats[round_idx].items():
            eval_log[f"Eval_{dataset_name}/Round_{round_idx}/{key}"] = value
        for key, value in round_improve_stats[round_idx].items():
            eval_log[f"Eval_{dataset_name}/Round_{round_idx}/{key}"] = value
        if round_idx > 0:
            round_improve_rewards[round_idx] = torch.tensor(
                round_improve_rewards[round_idx], dtype=torch.float32, device="cuda"
            )
            dist.all_reduce(round_improve_rewards[round_idx], op=dist.ReduceOp.SUM)
            eval_log[f"Eval_{dataset_name}/Round_{round_idx}_vs_{round_idx-1}"] = (
                round_improve_rewards[round_idx].item()
            )
    for key, value in finish_stats.items():
        value = torch.tensor(value, device="cuda", dtype=torch.float)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        eval_log[f"Eval_{dataset_name}/{key}"] = value.item() / validation_set_len
    return eval_log


evaluate_geneval_raw_multi_round = evaluate_multi_round
evaluate_tiif_raw_multi_round = evaluate_multi_round


def save_inference_results(inference_results, curr_step, results_dir, mode="train", num_samples_to_save=2, timestep_idx=None, last_reward=None, change_summary=None):
    from train.rollout_controller import RolloutStepResult
    save_list = []
    per_sample_results = inference_results["per_sample_results"]
    per_sample_rewards = inference_results["per_sample_rewards"]
    if last_reward is not None:
        save_list.append({"last_reward": last_reward})
    if change_summary is not None and len(change_summary) > 0:
        save_list.append({"change_summary": change_summary})
    if timestep_idx is not None:
        save_list.append({f"sde_timestep_idx": timestep_idx})
    if "summary" in inference_results:
        per_sample_summary = inference_results["summary"]
        for sample_idx in range(len(per_sample_summary)):
            save_list.append({f"summary_{sample_idx}": per_sample_summary[sample_idx]})
    for sample_idx in range(min(num_samples_to_save, len(per_sample_results))):
        for step_result_list in per_sample_results[sample_idx]:
            step_result_list: List[RolloutStepResult]
            for step_result in step_result_list:
                step_result_dict = step_result.to_dict()
                if step_result_dict['is_cfg_text'] or step_result_dict['is_cfg_img']:
                    continue
                sample_idx = step_result.sample_idx
                round_idx = step_result.round_idx
                reward = per_sample_rewards[sample_idx][round_idx]
                step_result_dict["reward"] = reward
                save_list.append(step_result_dict)

    save_result_as_html(save_list, curr_step, results_dir, mode=mode)



class RolloutBuffer:
    def __init__(self, max_size=10000):
        self.perfect_buffer = []
        self.bad_buffer = []
        self.max_size = max_size
    
    def update(self, metadata, reward, img):
        if reward == 1:
            self.perfect_buffer.append((metadata, reward, img))
        else:
            self.bad_buffer.append((metadata, reward, img))

    def all_gather(self):
        all_perfect_buffer = [None] * dist.get_world_size()
        all_bad_buffer = [None] * dist.get_world_size()
        dist.all_gather_object(all_perfect_buffer, self.perfect_buffer)
        dist.all_gather_object(all_bad_buffer, self.bad_buffer)
        dist.barrier()
        gathered_perfect_buffer = []
        gathered_bad_buffer = []
        for perfect_buffer in all_perfect_buffer:
            gathered_perfect_buffer.extend(perfect_buffer)
        for bad_buffer in all_bad_buffer:
            gathered_bad_buffer.extend(bad_buffer)
        self.perfect_buffer = gathered_perfect_buffer
        self.bad_buffer = gathered_bad_buffer
        torch.cuda.empty_cache()
        
    def clear(self):
        self.perfect_buffer = []
        self.bad_buffer = []

    def __len__(self):
        return (len(self.perfect_buffer), len(self.bad_buffer))
    
    def sample(self, seed:int=0, perfect_ratio=0.15):
        perfect_len, bad_len = len(self.perfect_buffer), len(self.bad_buffer)
        g=torch.Generator(device="cuda")
        g.manual_seed(seed)
        prob = torch.tensor([perfect_ratio], device="cuda")
        mode = torch.bernoulli(prob, generator=g).item()
        if perfect_len == 0: mode = 0
        if bad_len == 0: mode = 1
        if mode == 0:
            idx = torch.randint(0, bad_len, (1,), generator=g).item()
            return self.bad_buffer[idx]
        else:
            idx = torch.randint(0, perfect_len, (1,), generator=g).item()
            return self.perfect_buffer[idx]



def main():
    dist.init_process_group("nccl", timeout=timedelta(seconds=1000))
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    args: tuple[ModelArguments, DataArguments, TrainingArguments] = (
        parser.parse_args_into_dataclasses()
    )
    model_args, data_args, training_args = args

    training_args.cfg_interval = (training_args.cfg_interval_low, training_args.cfg_interval_high)
    if not training_args.exp_name:
        training_args.exp_name = training_args.wandb_name

    if training_args.log_dir:
        training_args.log_dir = os.path.abspath(training_args.log_dir)
        training_args.mydir = training_args.log_dir
        training_args.results_dir = os.path.join(
            training_args.log_dir,
            training_args.wandb_project,
            training_args.exp_name,
        )
        training_args.checkpoint_dir = os.path.join(
            training_args.results_dir,
            "ckpts",
        )
    else:
        training_args.checkpoint_dir = os.path.abspath(
            os.path.join(training_args.mydir, training_args.checkpoint_dir)
        )
        training_args.results_dir = os.path.abspath(
            os.path.join(training_args.mydir, training_args.results_dir)
        )

    if dist.get_rank() == 0:
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        os.makedirs(training_args.results_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project,
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}",
            name=training_args.wandb_name,
            mode="offline" if training_args.wandb_offline else "online",
        )
        wandb.config.update(training_args, allow_val_change=True)
        wandb.config.update(model_args, allow_val_change=True)
        wandb.config.update(data_args, allow_val_change=True)
    else:
        logger = create_logger(None, dist.get_rank())
    logger.info(f"Training arguments {training_args}")
    logger.info(f"Model arguments {model_args}")
    logger.info(f"Data arguments {data_args}")
    logger.info(f"checkpoint_dir: {training_args.checkpoint_dir}")
    logger.info(f"results_dir: {training_args.results_dir}")

    def _format_reward_base_url(host: str, port: str) -> str:
        host = host.strip()
        port = port.strip()
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{port}/v1"

    if training_args.reward_server_port == "-1":
        # update reward server urls for using external service, e.g. openai, gemini, etc.
        reward_urls = [training_args.reward_server_urls]
    else:
        # update reward server urls for self deployed reward model
        reward_server_urls = training_args.reward_server_urls.split(",")
        reward_server_port = training_args.reward_server_port.split(",")
        reward_urls = []
        for url in reward_server_urls:
            for port in reward_server_port:
                reward_urls.append(_format_reward_base_url(url, port))
    logger.info(f"reward_urls: {reward_urls}")

    world_size = dist.get_world_size()
    if training_args.num_shard <= 0:
        raise ValueError(f"num_shard must be positive, got {training_args.num_shard}")
    normalized_sharding_strategy = (
        "FULL_SHARD"
        if training_args.sharding_strategy == "SHARD_GRAD_OP"
        else training_args.sharding_strategy
    )
    if normalized_sharding_strategy == "HYBRID_SHARD":
        if world_size % training_args.num_shard != 0:
            raise ValueError(
                f"World size {world_size} must be divisible by num_shard {training_args.num_shard}"
            )
        training_args.num_replicate = world_size // training_args.num_shard
    else:
        training_args.num_replicate = max(1, world_size // training_args.num_shard)
    logger.info(
        "Resolved FSDP2 mesh config: "
        f"world_size={world_size}, "
        f"sharding_strategy={training_args.sharding_strategy}, "
        f"num_shard={training_args.num_shard}, "
        f"num_replicate={training_args.num_replicate}, "
        "reshard_after_forward=False"
    )

    # Setup policy groups if enabled
    policy_groups = None
    if data_args.policy_group_size > 1:
        world_size = dist.get_world_size()
        if world_size % data_args.policy_group_size != 0:
            raise ValueError(
                f"World size {world_size} must be divisible by policy_group_size {data_args.policy_group_size}"
            )

        policy_group_id = dist.get_rank() // data_args.policy_group_size
        policy_group_rank = dist.get_rank() % data_args.policy_group_size

        # createpolicy groups
        policy_groups = []
        for i in range(world_size // data_args.policy_group_size):
            group_ranks = list(
                range(
                    i * data_args.policy_group_size,
                    (i + 1) * data_args.policy_group_size,
                )
            )
            group = dist.new_group(ranks=group_ranks)
            policy_groups.append(group)

        current_policy_group = policy_groups[policy_group_id]

        if dist.get_rank() == 0:
            logger.info(
                f"Created {len(policy_groups)} policy groups with size {data_args.policy_group_size}"
            )
            logger.info(f"Rank {dist.get_rank()} is in policy group {policy_group_id}")
    else:
        current_policy_group = None
        policy_group_id = dist.get_rank()
        policy_group_rank = 0

    # auto resume from latest checkpoint
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model:
    llm_config:Qwen2Config = Qwen2Config.from_pretrained(model_args.llm_path)
    if training_args.debug:
        llm_config.num_hidden_layers = 1
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    language_model = Qwen2ForCausalLM(llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:
        vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = (
            vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        )
        vit_config.rope = model_args.vit_rope
        vit_model = SiglipVisionModel(vit_config)

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=model_args.vae_path,
        )

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(
        language_model, vit_model if training_args.visual_und else None, config
    )

    patch_latent_dim = model.patch_latent_dim

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(os.path.dirname(model_args.llm_path))
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    # Setup FSDP2 and load pretrained model:
    num_shard = training_args.num_shard
    num_replicate = training_args.num_replicate
    fsdp_config = FSDP2Config(
        sharding_strategy=training_args.sharding_strategy,
        cpu_offload=training_args.cpu_offload,
        num_replicate=num_replicate,
        num_shard=num_shard,
    )
    ema_model = None
    if not training_args.debug:
        model, ema_model = FSDPCheckpoint.try_load_ckpt(
            resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
        )

    if training_args.kl_weight_text > 0 or training_args.kl_weight_image > 0:
        ref_model = deepcopy(model)
        ref_model.requires_grad_(False)
        ref_model.eval()
        ref_model = apply_fsdp2_with_activation_checkpointing(ref_model, fsdp_config, use_ac=False)
        fsdp2_lazy_init_root(ref_model)
    else:
        ref_model = None

    fsdp_model = apply_fsdp2_with_activation_checkpointing(model, fsdp_config, use_ac=True)
    fsdp2_lazy_init_root(fsdp_model)

    # Setup optimizer and scheduler
    params_to_optimize = list(fsdp_model.parameters())
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=training_args.lr,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=training_args.weight_decay,
    )

    if training_args.lr_scheduler == "cosine":
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0 if training_args.resume_step == -1 else training_args.resume_step
        data_status = None
    else:
        if resume_from is not None and os.path.isdir(resume_from) and os.path.exists(os.path.join(resume_from, ".metadata")):
            optimizer, scheduler, train_step, data_status = FSDP2Checkpoint.load_training_state(
                resume_from,
                fsdp_model,
                optimizer,
                scheduler,
                fsdp_config,
            )
        else:
            optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
                resume_from,
                optimizer,
                scheduler,
                fsdp_config,
            )

    # setup dataset
    if data_args.dataset_name in ["geneval", "geneval_plus"]:
        train_dataset = GenevalPromptDataset(data_args.train_data_path)
        test_dataset = GenevalPromptDataset(data_args.val_data_path)
    elif data_args.dataset_name == "tiif":
        train_dataset = TIIFDataset(data_args.train_data_path)
        test_dataset = TIIFDataset(data_args.val_data_path)
    else:
        raise ValueError(f"Invalid dataset name: {data_args.dataset_name}")
    # 创建无限循环的DataLoader
    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=1,
        k=1,
        world_size=dist.get_world_size(),
        rank=policy_group_id, # make sure each group has the same data
        seed=42,
    )
    # 创建DataLoader，注意这里不需要shuffle，由Sampler控制
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=train_dataset.collate_fn,
    )
    train_iter = iter(train_dataloader)

    vae_transform = ImageTransform(
        max_image_size=data_args.vae_max_image_size,
        min_image_size=data_args.vae_min_image_size,
        image_stride=data_args.vae_image_stride,
    )
    vit_transform = ImageTransform(
        max_image_size=data_args.vit_max_image_size,
        min_image_size=data_args.vit_min_image_size,
        image_stride=data_args.vit_image_stride,
    )

    if training_args.visual_gen:
        vae_model.to(device).eval()

    torch.set_default_device("cuda")

    # 创建reward函数
    executor = futures.ThreadPoolExecutor(max_workers=8)
    if training_args.debug:
        reward_fn, eval_reward_fn = build_debug_reward_fns(training_args.reward_fn)
        logger.info("Debug mode enabled: using local stub reward functions that always return zero.")
    elif training_args.reward_fn == "geneval":
        reward_fn = geneval_score(reward_urls[dist.get_rank() % len(reward_urls)])
        eval_reward_fn = geneval_score(reward_urls[dist.get_rank() % len(reward_urls)])
    elif training_args.reward_fn == "geneval_plus":
        reward_fn = geneval_plus_train_reward(url=reward_urls[dist.get_rank() % len(reward_urls)], model_name=training_args.reward_model_name, api_key=training_args.reward_api_key, client_type=training_args.client_type)
        eval_reward_fn = geneval_plus_eval_reward(url=reward_urls[dist.get_rank() % len(reward_urls)], model_name=training_args.reward_model_name, api_key=training_args.reward_api_key)
    elif training_args.reward_fn == "tiif":
        reward_fn = vqa_reward_fn(url=reward_urls[dist.get_rank() % len(reward_urls)], model_name=training_args.reward_model_name, api_key=training_args.reward_api_key, client_type=training_args.client_type)
        eval_reward_fn = tiif_reward_fn(url=reward_urls[dist.get_rank() % len(reward_urls)], model_name=training_args.reward_model_name, api_key=training_args.reward_api_key,)
    else:
        raise ValueError(f"Invalid reward function: {training_args.reward_fn}")

    # Prepare models for training:
    fsdp_model.train()

    # Online RL training loop
    logger.info(
        f"Starting online RL training for {training_args.total_steps} steps, starting at {train_step}..."
    )

    # prepare rollout kwargs
    image_shape = (training_args.image_size, training_args.image_size)
    generation_kwargs = dict(
        think=training_args.think_mode,
        max_output_token_n_gen=training_args.max_output_token_n_gen,
        max_output_token_n_edit=training_args.max_output_token_n_edit,
        do_sample=training_args.do_sample,
        text_temperature=training_args.text_temperature,
        topk=training_args.top_k,
        top_p=training_args.top_p,
        use_static_cache=training_args.use_static_kv_cache,
        cfg_text_scale=training_args.cfg_text_scale,
        cfg_img_scale=training_args.cfg_img_scale,
        cfg_interval=training_args.cfg_interval,
        timestep_shift=training_args.timestep_shift,
        num_timesteps_gen=training_args.num_timesteps_gen,
        num_timesteps_edit=training_args.num_timesteps_edit,
        cfg_renorm_min=training_args.cfg_renorm_min,
        cfg_renorm_type_gen=training_args.cfg_renorm_type_gen,
        cfg_renorm_type_edit=training_args.cfg_renorm_type_edit,
        image_shapes = image_shape,
        executor=executor,
        reward_fn=reward_fn,
        reward_fn_type=training_args.reward_fn,
        enable_sde=training_args.tune_image_cot,
    )
    infer_model = fsdp_model
    if training_args.tune_image_cot:    
        sde_sampler_gen = SDESampler(eta_mode=training_args.eta_mode, constant_eta=training_args.constant_eta_gen, model_output_type="velocity")
        sde_sampler_edit = SDESampler(eta_mode=training_args.eta_mode, constant_eta=training_args.constant_eta_edit, model_output_type="velocity")
    else:
        sde_sampler_gen = None
        sde_sampler_edit = None
    logger.info(f"Using SDESampler Gen: {sde_sampler_gen} \n SDESampler Edit: {sde_sampler_edit}")
    rollout_controller = MultiRoundRolloutController(
        model=infer_model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
        sde_sampler=sde_sampler_gen,
    )
    rollout_buffer = RolloutBuffer()

    for curr_step in range(train_step, training_args.total_steps):
        if training_args.debug and curr_step == train_step:
            logger.info("Debug mode enabled: skipping initial evaluation loop.")
        elif curr_step % training_args.eval_freq == 0:
            infer_model.eval()
            eval_generation_kwargs = generation_kwargs.copy()
            eval_generation_kwargs["enable_sde"] = False
            eval_generation_kwargs["num_timesteps_gen"] = 50
            eval_generation_kwargs["num_timesteps_edit"] = 50
            eval_generation_kwargs["text_temperature"] = 0.3
            eval_generation_kwargs["reward_fn"] = eval_reward_fn
            eval_generation_kwargs["reward_fn_type"] = training_args.reward_fn
            eval_log = evaluate_multi_round(
                test_dataset, rollout_controller, training_args, data_args, curr_step,
                generation_kwargs=eval_generation_kwargs, eval_rounds=training_args.eval_rounds,
            )
            logger.info(f"eval_log: {eval_log}")
            if dist.get_rank() == 0:
                for key, value in eval_log.items():
                    logger.info(f"Rank {dist.get_rank()}: {key}: {value}")
                wandb.log(eval_log, step=curr_step)
        if curr_step >= training_args.total_steps: break

        torch.cuda.synchronize()
        start_time = time.time()
        group_seed = (policy_group_id + 1) * training_args.total_steps + curr_step
        gen_seed = (dist.get_rank() + 1) * training_args.total_steps + curr_step
        group_generator = torch.Generator(device="cuda")
        group_generator.manual_seed(group_seed)
        num_image_tokens = image_shape[0] * image_shape[1] // 16**2
        sample_generator = torch.Generator(device="cuda")
        sample_generator.manual_seed(gen_seed)
        rollout_batch_size = max(training_args.group_size // data_args.policy_group_size, 1)

        generation_kwargs["initial_noise"] = [torch.randn(
            num_image_tokens, patch_latent_dim, dtype=torch.bfloat16, generator=group_generator, device="cuda"
        ) for _ in range(2)]

        with rollout_controller.context():
            if curr_step % training_args.rounds == 0:
                # Round 1: reasoning stage
                rollout_buffer.clear()
                train_sampler.set_epoch(curr_step)
                prompts, metadatas = next(train_iter)
                metadata = metadatas[0]
                generation_kwargs["enable_sde"] = training_args.enable_sde_gen
                if training_args.enable_sde_gen:
                    # sample timestep_idx from uniform distribution
                    timestep_idx = torch.randint(0, training_args.num_timesteps_gen - 1, (1,), device="cuda", dtype=torch.int, generator=group_generator)
                    generation_kwargs["sde_timestep_idx"] = timestep_idx.tolist()
                    rollout_controller.sde_sampler = sde_sampler_gen
                inference_results = rollout_controller.rollout(
                    rounds=1,
                    prompts = prompts * rollout_batch_size,
                    prompt_metadata = metadatas * rollout_batch_size,
                    generator=sample_generator,
                    **generation_kwargs,
                )
            else:
                # Round 2: reflect and refine stage
                metadata, reward, img = rollout_buffer.sample(seed=group_seed)
                last_reward = reward
                prompt = metadata['prompt']
                reflection_system_prompt = '''The description of the target image is: {prompt}\nHow to further edit the provided image to make it consistent with the target description? Please provide concrete editing instructions in a single sentence. If no editing operation is needed, answer: no further edit needed.'''
                reflection_system_prompt = reflection_system_prompt.format(prompt=prompt)
                rollout_batch_size = max(training_args.group_size // data_args.policy_group_size, 1)
                generation_kwargs["enable_sde"] = training_args.enable_sde_edit
                if training_args.enable_sde_edit:
                    # sample timestep_idx from uniform distribution
                    timestep_idx = torch.randint(0, training_args.max_sde_timestep_idx_for_edit, (1,), device="cuda", dtype=torch.int, generator=group_generator)
                    generation_kwargs["sde_timestep_idx"] = timestep_idx.tolist()
                    rollout_controller.sde_sampler = sde_sampler_edit
                inference_results = rollout_controller.rollout(
                    rounds=2,
                    start_round_idx=1,
                    prompts = [prompt] * rollout_batch_size,
                    reflection_prompts = [reflection_system_prompt] * rollout_batch_size,
                    prompt_metadata = [metadata] * rollout_batch_size,
                    input_imgs = [img] * rollout_batch_size,
                    generator=sample_generator,
                    **generation_kwargs,
                )
        # first get all rewards, get per round rewards and cot length for logging 
        inference_results['per_sample_total_rewards'] = [[] for _ in range(rollout_batch_size)]
        edit_img_adv_multiplier = []
        per_round_format_rewards = [[] for _ in range(1)]
        for sample_idx, per_sample_reward in enumerate(inference_results["per_sample_rewards"]):
            for round_idx, per_round_reward in enumerate(per_sample_reward):
                if training_args.reward_fn == "geneval":
                    all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict = per_round_reward.result()
                    round_results = all_scores
                elif training_args.reward_fn == "geneval_plus":
                    round_results = per_round_reward.result()
                elif training_args.reward_fn == "tiif":
                    round_results = (per_round_reward.result(), "")
                per_sample_reward[round_idx] = round_results[0]
                per_round_format_rewards[round_idx].append(inference_results['per_sample_format_rewards'][sample_idx][round_idx])
                if curr_step % training_args.rounds == 0:
                    format_reward = inference_results['per_sample_format_rewards'][sample_idx][round_idx]
                    total_reward = round_results[0] * format_reward + training_args.format_reward_weight * format_reward
                    inference_results['per_sample_total_rewards'][sample_idx].append(total_reward)
                    # update rollout buffer
                    rollout_buffer.update(metadata, round_results[0], inference_results['output_imgs'][sample_idx])
                else:
                    this_reward = round_results[0]
                    improvement = this_reward - last_reward

                    format_reward = inference_results['per_sample_format_rewards'][sample_idx][round_idx]
                    if last_reward == 1 and "no further edit" in inference_results['per_sample_edit_operations'][sample_idx].lower():
                        improvement = 1
                    if "no further edit" in inference_results['per_sample_edit_operations'][sample_idx].lower():
                        edit_img_adv_multiplier.append(0)
                    else:
                        edit_img_adv_multiplier.append(1)

                    total_reward = format_reward * training_args.format_reward_weight + improvement * format_reward
                    inference_results['per_sample_total_rewards'][sample_idx].append(total_reward)

        if curr_step % training_args.rounds == 0:
            rollout_buffer.all_gather()
        logger.info(f"Rank {dist.get_rank()}: Rollout buffer length: {rollout_buffer.__len__()}")
        # compute advantages
        text_advantages, additional_stats = compute_advantages_for_multi_round_rollout(
            inference_results["per_sample_total_rewards"], # with text format reward
            current_policy_group,
            first_subtract_mean=True,
            first_divide_std=False,
            later_operations="",
            first_scaler=1.0,
            later_scaler=1.0,
        )
        image_advantages, additional_stats = compute_advantages_for_multi_round_rollout(
            inference_results["per_sample_rewards"], # without text format reward
            current_policy_group,
            first_subtract_mean=True,
            first_divide_std=False,
            later_operations="",
            first_scaler=1.0,
            later_scaler=1.0,
        )

        if curr_step % training_args.rounds != 0:
            # mask image advantage if no further edit
            for sample_idx, multiplier in enumerate(edit_img_adv_multiplier):
                image_advantages[sample_idx][0] = image_advantages[sample_idx][0] * multiplier
        logger.info(f"Rank {dist.get_rank()}: Text advantages: {text_advantages}")
        logger.info(f"Rank {dist.get_rank()}: Image advantages: {image_advantages}")

        # update rollout stats
        round_idx = curr_step % training_args.rounds
        mode = "gen" if curr_step % training_args.rounds == 0 else "edit"
        mode = mode + "_" + str(round_idx)
        # process cot length
        round_mean_cot_length = inference_results[f'Round_0/mean_cot_length']
        round_mean_cot_length = torch.tensor(round_mean_cot_length, device="cuda", dtype=torch.float).mean()
        dist.all_reduce(round_mean_cot_length, op=dist.ReduceOp.AVG)
        additional_stats[f'{mode}/mean_cot_length'] = round_mean_cot_length.item()
        # process format reward
        round_mean_format_reward = torch.tensor(per_round_format_rewards[0], device="cuda", dtype=torch.float).mean()
        dist.all_reduce(round_mean_format_reward, op=dist.ReduceOp.AVG)
        additional_stats[f'{mode}/mean_format_reward'] = round_mean_format_reward.item()
        additional_stats[f'{mode}/mean_reward'] = additional_stats['round_mean_rewards'].pop("round_0")


        dist.barrier()
        loss_dict = {
            "text_policy_loss": [],
            "img_policy_loss": [],
            "text_kl_loss": [],
            "img_kl_loss": [],
            "text_prob_ratio_mean": [],
            "text_prob_ratio_max": [],
            "text_prob_ratio_min": [],
            "img_prob_ratio_mean": [],
            "img_prob_ratio_max": [],
            "img_prob_ratio_min": [],
        }
        fsdp_model.train()
        optimizer.zero_grad()

        # NOTE: We collect all batches to ensure synchronized iteration counts across ranks.
        # This uses more memory but avoids doing the expensive pack_iterator work twice.
        # The pack_iterator does heavy tensor operations, tokenization, and data packing,
        # so materializing results once is more efficient than iterating twice.
        (
            all_data_points,
            total_text_grad_tokens,
            total_image_num,
        ) = rollout_controller.prepack(
            inference_results,
            with_text_loss=training_args.tune_text_cot,
            with_img_loss=training_args.tune_image_cot,
            timestep_ratio=training_args.timestep_sample_ratio,
            pack_start_round_idx=training_args.pack_start_round_idx,
            pack_end_round_idx=training_args.rounds-1,
        )

        logger.info(
            f"Rank {dist.get_rank()}: Processing {len(all_data_points)} iterations"
        )
        logger.info(f"Rank {dist.get_rank()}: total text grad tokens: {total_text_grad_tokens}, total image num: {total_image_num}")

        # Process the synchronized number of batches

        for iteration_count, (packed_input, loss_info) in enumerate(
            tqdm(
                rollout_controller.pack_iterator(
                    all_data_points,
                    text_advantages=text_advantages,
                    image_advantages=image_advantages,
                    cfg_interval=training_args.cfg_interval,
                ),
                total=1,
                desc=f"Gradient Accumulation [{total_text_grad_tokens} Text Tokens, {total_image_num} Images]",
            )
        ):
            # repeat advantages to match the length of label_ids
            cur_text_advantages = loss_info.get("text_advantages", None)
            cur_image_advantages = loss_info.get("image_advantages", None)
            behavior_probs = loss_info.get("inference_probs", None)
            label_ids = loss_info.get("packed_label_ids", None)
            # Step 4: training model with the packed data
            # Forward pass with mixed precision
            loss = 0
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                # Compute ref model logits
                with torch.inference_mode():
                    if (training_args.kl_weight_text > 0 or training_args.kl_weight_image > 0): # forward ref model for text kl loss
                        ref_loss_dict = ref_model(**packed_input, ref_forward=True)
                        ref_per_token_logps = (
                            ref_loss_dict["logits"]
                            .log_softmax(dim=-1)
                            .gather(dim=-1, index=label_ids.unsqueeze(-1))
                            .squeeze(-1)
                        )
                        logger.info(
                            f"Rank {dist.get_rank()}: Ref per token logps shape: {ref_per_token_logps.shape}"
                        )
                    if (training_args.kl_weight_image > 0 and training_args.tune_image_cot):
                        ref_v_pred = ref_loss_dict["packed_mse_preds"]

                output_dict = fsdp_model(**packed_input)
            if training_args.tune_text_cot:
                raw_per_token_logps = (
                    output_dict["logits"]
                    .log_softmax(dim=-1)
                    .gather(dim=-1, index=label_ids.unsqueeze(-1))
                    .squeeze(-1)
                )
                per_token_logps = gather_text_policy_logps(
                    output_dict["logits"],
                    label_ids,
                    do_sample=training_args.do_sample,
                    temperature=training_args.text_temperature,
                    topk=training_args.top_k,
                    top_p=training_args.top_p,
                )

                if behavior_probs is not None:
                    behavior_logps = behavior_probs.detach().clamp_min(1e-20).log()
                    ratio1 = torch.exp(per_token_logps - behavior_logps)
                    ratio2 = torch.clamp(
                        ratio1,
                        min=1.0 - training_args.grpo_clip_ratio,
                        max=1.0 + training_args.grpo_clip_ratio,
                    )
                    per_token_loss = -torch.minimum(
                        ratio1 * cur_text_advantages,
                        ratio2 * cur_text_advantages,
                    ).sum() / (1e-4 + total_text_grad_tokens)
                    loss_dict["text_prob_ratio_mean"].append(ratio1.mean().item())
                    loss_dict["text_prob_ratio_max"].append(ratio1.max().item())
                    loss_dict["text_prob_ratio_min"].append(ratio1.min().item())
                else:
                    per_token_loss = -(
                        torch.exp(per_token_logps - per_token_logps.detach())
                        * cur_text_advantages
                    ).sum() / (1e-4 + total_text_grad_tokens)
                loss_dict["text_policy_loss"].append(per_token_loss.item())
                if training_args.kl_weight_text > 0:
                    per_token_kl = (
                        torch.exp(ref_per_token_logps - raw_per_token_logps)
                        - (ref_per_token_logps - raw_per_token_logps)
                        - 1
                    )  # k3 estimation
                    per_token_kl = per_token_kl.sum() / (1e-4 + total_text_grad_tokens)
                    logger.info(
                        f"Rank {dist.get_rank()}: Text kl loss: {per_token_kl.item()}"
                    )
                    loss_dict["text_kl_loss"].append(per_token_kl.item())
                    # ignore unexpected large kl loss
                    if per_token_kl.item() > 10:
                        per_token_kl = per_token_logps.mean() * 0.0
                    loss = per_token_loss + training_args.kl_weight_text * per_token_kl
                else:
                    loss = per_token_loss
                loss = loss * training_args.ce_weight
                logger.info(f"Rank {dist.get_rank()}: Text Loss: {loss.item()}")

            v_pred = output_dict["packed_mse_preds"]
            if training_args.tune_image_cot and v_pred is not None:
                if curr_step % training_args.rounds == 0:
                    sde_sampler = sde_sampler_gen
                else:
                    sde_sampler = sde_sampler_edit
                v_pred = rollout_controller.apply_cfg(
                        v_pred,
                        loss_info["mse_condition_token_indexes"],
                        training_args.cfg_text_scale,
                        training_args.cfg_img_scale,
                        training_args.cfg_interval,
                        training_args.cfg_renorm_min,
                        training_args.cfg_renorm_type_gen,
                        training_args.cfg_renorm_type_edit,
                    )
                logger.info(f"Rank {dist.get_rank()}: v_pred shape: {v_pred.shape}")
                x_curr = sde_sampler.get_x_t_distribution(
                    loss_info["packed_noisy_latent_for_loss"].float(),
                    loss_info["packed_timesteps_for_loss"].float(),
                    loss_info["dts"].float(),
                    v_pred.float(),
                )
                x_prev = loss_info["packed_prev_latents"]
                log_prob = x_curr.log_prob(x_prev)
                # convert it to image level loss
                skip_interval = 1
                img_latent_lengths = [h * w for h, w in loss_info['patchified_vae_latent_shapes_for_loss'][::skip_interval]] # ::2 to skip unconditioned images
                assert sum(img_latent_lengths) == log_prob.shape[0]
                log_prob_per_image = torch.stack([x.mean() for x in torch.split(log_prob, img_latent_lengths)]) # (num_images,)
                logger.info(f"Rank {dist.get_rank()}: log_prob_per_image: {log_prob_per_image}")
                behavior_log_prob = loss_info.get("log_probs", None)
                if behavior_log_prob is not None:
                    behavior_log_prob_per_image = torch.stack(
                        [x.mean() for x in torch.split(behavior_log_prob.float(), img_latent_lengths)]
                    )
                    ratio1 = torch.exp(log_prob_per_image - behavior_log_prob_per_image.detach())
                    ratio2 = torch.clamp(
                        ratio1,
                        min=1.0 - training_args.grpo_clip_ratio,
                        max=1.0 + training_args.grpo_clip_ratio,
                    )
                    img_per_token_loss = -torch.minimum(
                        ratio1 * cur_image_advantages.to(log_prob_per_image.dtype),
                        ratio2 * cur_image_advantages.to(log_prob_per_image.dtype),
                    ).sum() / (1e-4 + total_image_num)
                    loss_dict["img_prob_ratio_mean"].append(ratio1.mean().item())
                    loss_dict["img_prob_ratio_max"].append(ratio1.max().item())
                    loss_dict["img_prob_ratio_min"].append(ratio1.min().item())
                else:
                    img_per_token_loss = -(
                        torch.exp(log_prob_per_image - log_prob_per_image.detach()) * cur_image_advantages.to(log_prob_per_image.dtype)
                    ).sum() / (1e-4 + total_image_num)
                loss_dict["img_policy_loss"].append(img_per_token_loss.item())

                img_loss = img_per_token_loss
                kl_img_loss = 0
                if training_args.kl_weight_image > 0:
                    ref_v_pred = rollout_controller.apply_cfg(
                        ref_v_pred,
                        loss_info["mse_condition_token_indexes"],
                        training_args.cfg_text_scale,
                        training_args.cfg_img_scale,
                        training_args.cfg_interval,
                        training_args.cfg_renorm_min,
                        training_args.cfg_renorm_type_gen,
                        training_args.cfg_renorm_type_edit,
                    )
                    ref_x_curr = sde_sampler.get_x_t_distribution(
                        loss_info["packed_noisy_latent_for_loss"].float(),
                        loss_info["packed_timesteps_for_loss"].float(),
                        loss_info["dts"].float(),
                        ref_v_pred.float(),
                    )
                    kl_img_loss = x_curr.kl_divergence(ref_x_curr).mean()
                    loss_dict["img_kl_loss"].append(kl_img_loss.item())

                logger.info(f"Rank {dist.get_rank()}: Image Loss: {img_loss.item():.4f}, KL: {kl_img_loss:.4f}")
                loss = loss + img_loss * training_args.mse_weight + kl_img_loss * training_args.kl_weight_image
            # Backward pass
            loss.backward()
        # Gradient clipping and optimization step
        total_norm = clip_grad_norm_fsdp2(
            [param for param in fsdp_model.parameters() if param.requires_grad],
            training_args.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()
        logger.info(f"Rank {dist.get_rank()}: Total norm: {total_norm.item()}")
        for k in list(loss_dict.keys()):
            v = loss_dict[k]
            if len(v) == 0:
                loss_dict.pop(k)
                continue
            v_sum = torch.tensor(sum(v), device=device)
            v_size = torch.tensor(len(v), device=device)
            dist.all_reduce(v_sum, op=dist.ReduceOp.AVG)
            dist.all_reduce(v_size, op=dist.ReduceOp.AVG)
            loss_dict[k] = v_sum / v_size
        if ema_model is not None:
            fsdp2_ema_update(ema_model, fsdp_model, decay=training_args.ema)

        # Log loss values for this batch
        if curr_step % training_args.log_every == 0:
            # Measure training speed
            torch.cuda.synchronize()
            end_time = time.time()
            steps_per_sec = training_args.log_every / (end_time - start_time)
            message = f"(step={curr_step:07d})  step time: {end_time - start_time:.2f}s"
            wandb_log = {}
            train_log_values = {
                f"train/{k}": v.item() if torch.is_tensor(v) else float(v)
                for k, v in loss_dict.items()
            }
            wandb_log["step_time"] = end_time - start_time
            # all gather rewards
            for info_key, info_value in additional_stats.items():
                if isinstance(info_value, dict):
                    for key_, value_ in info_value.items():
                        wandb_log[f"{info_key}/{key_}"] = value_
                        message += f"{info_key}/{key_}: {value_:.4f}, "
                else:
                    wandb_log[info_key] = info_value
                    message += f"{info_key}: {info_value:.4f}, "

            for key, value in train_log_values.items():
                wandb_log[key] = value
                message += f"{key}: {value:.4f}, "

            message += f"Train Steps/Sec: {steps_per_sec:.2f}, "
            logger.info(message)

            wandb_log["lr"] = optimizer.param_groups[0]["lr"]
            wandb_log["total_norm"] = total_norm.item()

            mem_allocated = torch.tensor(
                torch.cuda.max_memory_allocated() / 1024**3, device=device
            )
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log["mem_allocated"] = mem_allocated
            mem_cache = torch.tensor(
                torch.cuda.max_memory_reserved() / 1024**3, device=device
            )
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log["mem_cache"] = mem_cache
            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)

        if curr_step > 0 and (curr_step % training_args.save_every == 0 or curr_step == training_args.total_steps - 1):
            try:
                FSDP2Checkpoint.save_checkpoint(
                    ckpt_dir=training_args.checkpoint_dir,
                    train_steps=curr_step,
                    model=fsdp_model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    logger=logger,
                    fsdp_config=fsdp_config,
                    data_status=None,  # online training不需要data_status
                )
            except Exception as e:
                logger.error(f"Rank {dist.get_rank()}: Save checkpoint failed: {e}")

    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
