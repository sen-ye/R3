# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from concurrent.futures import Executor
from typing import List, Dict, Tuple, Optional, Union, Any, Callable    
from contextlib import contextmanager
from PIL import Image
import numpy as np
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from data.data_utils import (
    get_flattened_position_ids_extrapolate, get_flattened_position_ids_interpolate, 
    patchify,
)
from modeling.bagel.qwen2_navit import NaiveCache, NaiveCacheMultiSeq
from modeling.bagel.bagel import Bagel
import random
import re
from dataclasses import dataclass, field

VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here.'''


@dataclass
class RolloutStepResult:
    sample_idx: int = 0
    round_idx: int = 0
    step_idx: int = 0
    is_cfg_text: bool = False
    is_cfg_img: bool = False
    output_list: List[Union[str, Image.Image]] = field(default_factory=list)
    need_loss_list: List[bool] = field(default_factory=list)
    content_type_list: List[str] = field(default_factory=list)
    
    need_vae: bool = True
    need_vit: bool = True


    # only need when output_list contains image and need loss
    latent: Union[torch.Tensor, List[torch.Tensor]] = None
    log_prob: Union[torch.Tensor, List[torch.Tensor]] = None
    timestep: Union[torch.Tensor, List[torch.Tensor]] = None
    dt: Union[torch.Tensor, List[torch.Tensor]] = None
    prev_latent: Union[torch.Tensor, List[torch.Tensor]] = None


    def to_dict(self):
        return_dict = {
            "sample_idx": self.sample_idx,
            "round_idx": self.round_idx,
            "step_idx": self.step_idx,
            "is_cfg_text": self.is_cfg_text,
            "is_cfg_img": self.is_cfg_img,
        }
        for i, (output, need_loss) in enumerate(zip(self.output_list, self.need_loss_list)):
            return_dict[f"content_{i}"] = output
            return_dict[f"content_type_{i}"] = self.content_type_list[i]
            return_dict[f"need_loss_{i}"] = need_loss
        return return_dict



class MultiRoundRolloutController:
    def __init__(self, model:Bagel, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids, sde_sampler):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.sde_sampler = sde_sampler
        
    @property
    def unwrapped_model(self):
        if isinstance(self.model, FSDP):
            return self.model.module
        else:
            return self.model

    
    @contextmanager
    def context(self):
        torch.cuda.empty_cache()
        self.model.eval()
        with FSDP.summon_full_params(self.model,recurse=False,writeback=False):
            yield
        torch.cuda.empty_cache()
        
    def init_gen_context(self, batch_size:int = 1, multi_seq: bool = False,): 
        num_layers = self.model.config.llm_config.num_hidden_layers
        gen_context = {
            'kv_lens': [0] * batch_size,
            'ropes': [0] * batch_size,
            'past_key_values': NaiveCacheMultiSeq(num_layers, batch_size) if multi_seq else NaiveCache(num_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.unwrapped_model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=text if isinstance(text, list) else [text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )
        selected_cache_indices = torch.arange(0, len(kv_lens), device=self.model.device)

        past_key_values = self.unwrapped_model.forward_cache_update_text(past_key_values, 
                                                               selected_cache_indices=selected_cache_indices, 
                                                               **generation_input)     
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 
        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.unwrapped_model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=image if isinstance(image, list) else [image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            selected_cache_indices = torch.arange(0, len(kv_lens), device=self.model.device)
            past_key_values = self.unwrapped_model.forward_cache_update_vae(self.vae_model, past_key_values, 
                            selected_cache_indices=selected_cache_indices, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.unwrapped_model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=image if isinstance(image, list) else [image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            selected_cache_indices = torch.arange(0, len(kv_lens), device=self.model.device)
            past_key_values = self.unwrapped_model.forward_cache_update_vit(past_key_values, 
                            selected_cache_indices=selected_cache_indices, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        batch_size: int = 1,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50, 
        timestep_shift=3.0,
        initial_noise=None,
        enable_sde=False,
        sde_timestep_idx=None,
    ):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.unwrapped_model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape] if isinstance(image_shape, tuple) else image_shape, 
            new_token_ids=self.new_token_ids,
            initial_noise=initial_noise
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.unwrapped_model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape] if isinstance(image_shape, tuple) else image_shape, 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.unwrapped_model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape] if isinstance(image_shape, tuple) else image_shape, 
        )

        if self.sde_sampler is None or not enable_sde:
            unpacked_latent = self.unwrapped_model.generate_image(
                past_key_values=past_key_values,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_img_past_key_values=cfg_img_past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                timestep_shift=timestep_shift,
                **generation_input,
                cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
                cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
                cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
                cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
                cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
                cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
                cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
                cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            )
            image = self.decode_image(unpacked_latent, image_shape, batch_size=batch_size)
            return image, [], [], [], []
        
        unpacked_latent, latents, log_probs, timesteps, dts = self.unwrapped_model.generate_image_mix(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            sde_sampler=self.sde_sampler,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            sde_timesteps_idx=sde_timestep_idx,
        )

        image = self.decode_image(unpacked_latent, image_shape, batch_size=batch_size)
        return image, latents, log_probs, timesteps, dts

    def decode_image(self, latent, image_shape, batch_size: int = 1):
        H, W = image_shape if isinstance(image_shape, tuple) else image_shape[0]
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample
        latent = torch.cat(latent, dim=0)
        latent = latent.reshape(batch_size, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(batch_size, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1) * 255
        images = []
        for i in range(batch_size):
            images.append((image[i]).to(torch.uint8).cpu().numpy())
        return images


    def get_think_format_reward(self, predict_str: str, must_end_with_think: bool=False) -> float:
        if not isinstance(predict_str, str):
            return 0.0
        predict_str = predict_str.strip()
        if must_end_with_think:
            pattern_str = r"^<think>(?:(?!<think>|</think>).)*?</think>$"
        else:
            pattern_str = r"^<think>(?:(?!<think>|</think>).)*?</think>(?:(?!<think>|</think>).)+$"

        pattern = re.compile(pattern_str, re.DOTALL)
        match_result = re.fullmatch(pattern, predict_str)

        return 1.0 if match_result else 0.0

    
    def extract_edit_operation(self, edit_prompts: List[str]) -> List[str]:
        edit_operations = []
        for prompt in edit_prompts:
            parts = prompt.split("</think>")
            if len(parts) > 1:
                edit_operations.append(parts[-1].strip())
            else:
                edit_operations.append("No further edit needed")
        return edit_operations
    
    
    def apply_cfg(self, v_pred, mse_condition_token_indexes, cfg_text_scale, cfg_img_scale, cfg_interval=(0.4, 1.0), cfg_renorm_min=0.0, cfg_renorm_type="global",
                  cfg_renorm_type_gen="global", cfg_renorm_type_edit="global"):
        cfg_combined_parts = []
        
        for token_indexes in mse_condition_token_indexes:
            v_t = v_pred[token_indexes[0][0]:token_indexes[0][1]]
            if len(token_indexes) > 1:
                cfg_text_v_t = v_pred[token_indexes[1][0]:token_indexes[1][1]]
            else:
                cfg_text_v_t = None
            if len(token_indexes) > 2:
                cfg_img_v_t = v_pred[token_indexes[2][0]:token_indexes[2][1]]
            else:
                cfg_img_v_t = None

            cfg_renorm_type = cfg_renorm_type_gen if len(token_indexes) == 1 else cfg_renorm_type_edit
            if cfg_text_v_t is not None and cfg_img_v_t is not None:
                if cfg_renorm_type == "text_channel":
                    v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                    scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                    v_t_text = v_t_text_ * scale
                    if cfg_img_scale > 1.0 and cfg_img_v_t is not None:
                        v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                    else:
                        v_t = v_t_text
                else:
                    v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                    
                    if cfg_img_scale > 1.0 and cfg_img_v_t is not None:
                        v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                    else:
                        v_t_ = v_t_text_

                    # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                    if cfg_renorm_type == "global":
                        norm_v_t = torch.norm(v_t)
                        norm_v_t_ = torch.norm(v_t_)
                    elif cfg_renorm_type == "channel":
                        norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                        norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                    else:
                        raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                    scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                    v_t = v_t_ * scale
            cfg_combined_parts.append(v_t)
        return torch.cat(cfg_combined_parts, dim=0)
    

    @torch.no_grad()
    def generate_image_with_think(
        self, 
        prompts: List[str],
        image_shapes, 
        round_idx: int = 0,
        think: bool = True,
        max_output_token_n: int = 256, 
        do_sample: bool = True,
        text_temperature: float = 0.3,
        topk: int = -1,
        cfg_text_scale=3.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        initial_noise=None,
        enable_sde: bool = False,
        generator: torch.Generator = None,
        sde_timestep_idx: List[int] = None,
    ):
        batch_size = len(prompts)
        batch_rollout_output = [[] for _ in range(batch_size)]
        extra_info = {}
        gen_context = self.init_gen_context(batch_size, multi_seq=True)
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        rollout_output_list = [RolloutStepResult(round_idx=round_idx, sample_idx=i) for i in range(batch_size)]
        cfg_rollout_output_list = [RolloutStepResult(round_idx=round_idx, is_cfg_text=True, sample_idx=i) for i in range(batch_size)]
        if think:
            system_prompt = GEN_THINK_SYSTEM_PROMPT
            gen_context = self.update_context_text([system_prompt] * batch_size, gen_context)
            cfg_text_context = deepcopy(gen_context)
            for i in range(batch_size):
                rollout_output_list[i].output_list.append(system_prompt)
                rollout_output_list[i].content_type_list.append("text")
                rollout_output_list[i].need_loss_list.append(False)
                if cfg_text_scale > 1.0:
                    cfg_rollout_output_list[i].output_list.append(system_prompt)
                    cfg_rollout_output_list[i].content_type_list.append("text")
                    cfg_rollout_output_list[i].need_loss_list.append(False)
        gen_context = self.update_context_text(prompts, gen_context)
        for i in range(batch_size):
            rollout_output_list[i].output_list.append(prompts[i])
            rollout_output_list[i].content_type_list.append("text")
            rollout_output_list[i].need_loss_list.append(False)
        if think:
            think_output_texts = []
            past_key_values: NaiveCacheMultiSeq = gen_context['past_key_values']
            kv_lens = gen_context['kv_lens']
            ropes = gen_context['ropes']
            generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
            generations, past_key_values = self.model.generate_text_batch(
                past_key_values=past_key_values,
                max_length=max_output_token_n,
                do_sample=do_sample,
                temperature=text_temperature,
                topk=topk,
                end_token_id=self.new_token_ids['eos_token_id'],
                generator=generator,
                **generation_input,
            )
            cot_lengths = [len(i) for i in generations]
            think_output_texts = self.tokenizer.batch_decode(generations, skip_special_tokens=False)
            extra_info[f"Round_{round_idx}/mean_cot_length"] = sum(cot_lengths) / batch_size
            per_sample_format_rewards = []
            for i, output_text in enumerate(think_output_texts):
                output_text = output_text.split('<|im_end|>')[0].split('<|im_start|>')[1]
                think_output_texts[i] = output_text
                per_sample_format_rewards.append(self.get_think_format_reward(output_text, must_end_with_think=True))
                ropes[i] += cot_lengths[i]
            extra_info[f"Round_{round_idx}/per_sample_format_rewards"] = per_sample_format_rewards
            kv_lens = past_key_values.get_seq_lens(range(batch_size))
            gen_context['kv_lens'] = kv_lens
            gen_context['ropes'] = ropes
            gen_context['past_key_values'] = past_key_values

            for i in range(batch_size):
                rollout_output_list[i].output_list.append(think_output_texts[i])
                rollout_output_list[i].content_type_list.append("text")
                rollout_output_list[i].need_loss_list.append(True)
        else:
            extra_info[f"Round_{round_idx}/per_sample_format_rewards"] = [0.0] * batch_size
            extra_info[f"Round_{round_idx}/mean_cot_length"] = 0.0

        img, latents, log_probs, timesteps, dts = self.gen_image(
            image_shapes, 
            gen_context, 
            batch_size=batch_size,
            cfg_text_precontext=cfg_text_context,
            cfg_img_precontext=cfg_img_context,
            cfg_text_scale=cfg_text_scale, 
            cfg_img_scale=1, 
            cfg_interval=cfg_interval, 
            timestep_shift=timestep_shift, 
            num_timesteps=num_timesteps,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            initial_noise=initial_noise,
            enable_sde=enable_sde,
            sde_timestep_idx=sde_timestep_idx,
        )
        for i in range(batch_size):
            rollout_output_list[i].output_list.append(Image.fromarray(img[i]))
            rollout_output_list[i].content_type_list.append("image_and_latent")
            rollout_output_list[i].need_loss_list.append(enable_sde)
            rollout_output_list[i].latent = latents[i] if enable_sde else None
            rollout_output_list[i].log_prob = log_probs[i] if enable_sde else None
            rollout_output_list[i].timestep = timesteps[i] if enable_sde else None
            rollout_output_list[i].dt = dts[i] if enable_sde else None
            
            batch_rollout_output[i].append(rollout_output_list[i])

            if cfg_text_scale > 1.0 and enable_sde:
                cfg_rollout_output_list[i].output_list.append(Image.fromarray(img[i]))
                cfg_rollout_output_list[i].content_type_list.append("image_and_latent")
                cfg_rollout_output_list[i].need_loss_list.append(enable_sde)
                cfg_rollout_output_list[i].latent = latents[i] if enable_sde else None
                cfg_rollout_output_list[i].log_prob = log_probs[i] if enable_sde else None
                cfg_rollout_output_list[i].timestep = timesteps[i] if enable_sde else None
                cfg_rollout_output_list[i].dt = dts[i] if enable_sde else None

                batch_rollout_output[i].append(cfg_rollout_output_list[i])

        return img, batch_rollout_output, extra_info
    

    @torch.no_grad()
    def get_edit_think_and_operation(self,
        round_idx: int = 1, 
        step_idx: int = 0,                             
        input_imgs: List[Image.Image] = [],
        prompts: List[str] = [],
        max_output_token_n: int = 256, 
        do_sample: bool = True,
        text_temperature: float = 0.3,
        topk: int = -1,
        use_vae_feature: bool = True,
        generator: torch.Generator = None,
        add_understand_think: bool = True,
    ):
        batch_size = len(input_imgs)
        batch_rollout_output = [[] for _ in range(batch_size)]
        extra_info = {}
        gen_context = self.init_gen_context(batch_size, multi_seq=True)
        if add_understand_think:
            system_prompt = VLM_THINK_SYSTEM_PROMPT
            gen_context = self.update_context_text([system_prompt] * batch_size, gen_context)
        gen_context = self.update_context_image(input_imgs, gen_context, vae=False, vit=True)
        gen_context = self.update_context_text(prompts, gen_context)
        past_key_values: NaiveCacheMultiSeq = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.unwrapped_model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        generations, past_key_values = self.unwrapped_model.generate_text_batch(
            past_key_values=past_key_values,
            max_length=max_output_token_n,
            do_sample=do_sample,
            temperature=text_temperature,
            topk=topk,
            end_token_id=self.new_token_ids['eos_token_id'],
            generator=generator,
            **generation_input,
        )
        cot_lengths = [len(i) for i in generations]
        extra_info[f"Round_{round_idx}/mean_cot_length"] = sum(cot_lengths) / batch_size
        per_sample_format_rewards = []
        think_output_texts = self.tokenizer.batch_decode(generations, skip_special_tokens=False)
        for i, output_text in enumerate(think_output_texts):
            output_text = output_text.split('<|im_end|>')[0].split('<|im_start|>')[1]
            think_output_texts[i] = output_text
            per_sample_format_rewards.append(self.get_think_format_reward(output_text, must_end_with_think=False))
        extra_info[f"Round_{round_idx}/per_sample_format_rewards"] = per_sample_format_rewards
        for i in range(batch_size):
            rollout_result = RolloutStepResult(round_idx=round_idx, step_idx=step_idx, sample_idx=i)
            if add_understand_think:
                rollout_result.output_list.append(system_prompt)
                rollout_result.content_type_list.append("text")
                rollout_result.need_loss_list.append(False)

            rollout_result.output_list.append(input_imgs[i])
            rollout_result.need_vae = False
            rollout_result.content_type_list.append("image")
            rollout_result.need_loss_list.append(False)

            rollout_result.output_list.append(prompts[i])
            rollout_result.content_type_list.append("text")
            rollout_result.need_loss_list.append(False)

            rollout_result.output_list.append(think_output_texts[i])
            rollout_result.content_type_list.append("text")
            rollout_result.need_loss_list.append(True)
            batch_rollout_output[i].append(rollout_result)

        return think_output_texts, batch_rollout_output, extra_info
    
    @torch.no_grad()
    def edit_image(self,
        round_idx: int = 1, 
        step_idx: int = 1,                             
        input_imgs: List[Image.Image] = [],
        edit_operations: List[str] = [],
        think: bool = False,
        max_output_token_n: int = 256, 
        do_sample: bool = True,
        text_temperature: float = 0.3,
        topk: int = -1,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        initial_noise=None,
        enable_sde: bool = False,
        sde_timestep_idx: torch.Tensor = None,
    ):
        batch_size = len(input_imgs)
        image_shapes = [input_imgs[0].size[::-1]] * batch_size
        batch_rollout_output = [[] for _ in range(batch_size)]
        extra_info = {}
        gen_context = self.init_gen_context(batch_size, multi_seq=True)
        cfg_img_context = deepcopy(gen_context)
        cfg_text_context = deepcopy(gen_context)

        gen_context = self.update_context_image(input_imgs, gen_context, vae=True, vit=True)
        gen_context = self.update_context_text(edit_operations, gen_context)

        if cfg_text_scale > 1.0:
            cfg_text_context = self.update_context_image(input_imgs, cfg_text_context, vae=True, vit=True)
        if cfg_img_scale > 1.0:
            cfg_img_context = self.update_context_text(edit_operations, cfg_img_context)

        img, latents, log_probs, timesteps, dts = self.gen_image(
            image_shapes, 
            gen_context, 
            batch_size=batch_size,
            cfg_text_precontext=cfg_text_context,
            cfg_img_precontext=cfg_img_context,
            cfg_text_scale=cfg_text_scale, 
            cfg_img_scale=cfg_img_scale, 
            cfg_interval=cfg_interval, 
            timestep_shift=timestep_shift, 
            num_timesteps=num_timesteps,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            initial_noise=initial_noise,
            enable_sde=enable_sde,
            sde_timestep_idx=sde_timestep_idx,
        )

        for i in range(batch_size):
            rollout_result = RolloutStepResult(round_idx=round_idx, step_idx=step_idx, sample_idx=i)
            rollout_result.output_list.append(input_imgs[i])
            rollout_result.content_type_list.append("image")
            rollout_result.need_loss_list.append(False)

            rollout_result.output_list.append(edit_operations[i])
            rollout_result.content_type_list.append("text")
            rollout_result.need_loss_list.append(False)

            rollout_result.output_list.append(Image.fromarray(img[i]))
            rollout_result.content_type_list.append("image_and_latent")
            rollout_result.need_loss_list.append(enable_sde)
            rollout_result.latent = latents[i] if enable_sde else None
            rollout_result.log_prob = log_probs[i] if enable_sde else None
            rollout_result.timestep = timesteps[i] if enable_sde else None
            rollout_result.dt = dts[i] if enable_sde else None

            batch_rollout_output[i].append(rollout_result)

            if cfg_text_scale > 1.0 and enable_sde:
                cfg_rollout_result = RolloutStepResult(round_idx=round_idx, step_idx=step_idx, is_cfg_text=True, sample_idx=i)
                cfg_rollout_result.output_list.append(input_imgs[i])
                cfg_rollout_result.content_type_list.append("image")
                cfg_rollout_result.need_loss_list.append(False)

                cfg_rollout_result.output_list.append(Image.fromarray(img[i]))
                cfg_rollout_result.content_type_list.append("image_and_latent")
                cfg_rollout_result.need_loss_list.append(enable_sde)
                cfg_rollout_result.latent = latents[i] if enable_sde else None
                cfg_rollout_result.log_prob = log_probs[i] if enable_sde else None
                cfg_rollout_result.timestep = timesteps[i] if enable_sde else None
                cfg_rollout_result.dt = dts[i] if enable_sde else None

                batch_rollout_output[i].append(cfg_rollout_result)
            
            if cfg_img_scale > 1.0 and enable_sde:
                cfg_rollout_result = RolloutStepResult(round_idx=round_idx, step_idx=step_idx, is_cfg_img=True, sample_idx=i)
                cfg_rollout_result.output_list.append(edit_operations[i])
                cfg_rollout_result.content_type_list.append("text")
                cfg_rollout_result.need_loss_list.append(False)

                cfg_rollout_result.output_list.append(Image.fromarray(img[i]))
                cfg_rollout_result.content_type_list.append("image_and_latent")
                cfg_rollout_result.need_loss_list.append(enable_sde)
                cfg_rollout_result.latent = latents[i] if enable_sde else None
                cfg_rollout_result.log_prob = log_probs[i] if enable_sde else None
                cfg_rollout_result.timestep = timesteps[i] if enable_sde else None
                cfg_rollout_result.dt = dts[i] if enable_sde else None

                batch_rollout_output[i].append(cfg_rollout_result)

        for i in range(batch_size):
            if "no further edit" in edit_operations[i].lower():
                img[i] = np.array(input_imgs[i])

        return img, batch_rollout_output, extra_info
    
    @torch.no_grad()
    def rollout_for_eval(
        self,
        rounds: int = 1,
        prompts: List[str] = None,
        reflection_prompts: List[str] = None,
        max_output_token_n_gen: int = 256, 
        max_output_token_n_edit: int = 512,
        do_sample: bool = True,
        text_temperature: float = 0.3,
        topk: int = -1,
        think: bool = True,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps_gen=50,
        num_timesteps_edit=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type_gen="global",
        cfg_renorm_type_edit="global",
        image_shapes: Tuple[int, int] = (1024, 1024),
        initial_noise: Optional[List[torch.Tensor]] = None,
        executor: Optional[Executor] = None,
        reward_fn: Callable = None, # reward function for each round, return a list of rewards for each sample
        reward_fn_type: str = "unified_reward", # unified_reward or geneval
        prompt_metadata: List[Dict] = None, # metadata for each prompt
        enable_sde: bool = False,
        generator: torch.Generator = None,
        sde_timestep_idx: List[int] = None,
        auto_stop: bool = True,
    ):
        batch_size = len(prompts)
        image_shapes = [image_shapes] * batch_size
        
        # Initialize output dictionary with multi-round structure
        output_dict = {
            'rounds': rounds,
            # per_sample_results: List of results for each sample, each sample_i has a list of n_round_i results
            'per_sample_results': [[] for _ in range(batch_size)],
            # per_sample_rewards: List of rewards for each sample, each sample_i has a list of n_round_i rewards
            'per_sample_rewards': [[] for _ in range(batch_size)],
            # per_sample_finished_steps: List of finished steps for each sample, each sample_i has a list of n_round_i finished steps
            'per_sample_finished_steps': [rounds] * batch_size,
            'per_sample_finished': [False] * batch_size,
            'per_sample_format_rewards': [[] for _ in range(batch_size)],
            'no_edit_cnt': [0 for _ in range(rounds)],
            'per_sample_imgs': [[] for _ in range(batch_size)],
        }

        active_seq_idx = list(range(batch_size))
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            current_images = [None] * batch_size  # Track current images for each sample
            
            for round_idx in range(rounds):
                if round_idx == 0:
                    img, batch_rollout_output, extra_info = self.generate_image_with_think(
                        round_idx=round_idx, 
                        prompts=prompts,
                        image_shapes=image_shapes,
                        think=think,
                        max_output_token_n=max_output_token_n_gen,
                        do_sample=do_sample,
                        text_temperature=text_temperature,
                        topk=topk,
                        cfg_text_scale=cfg_text_scale,
                        cfg_interval=cfg_interval,
                        timestep_shift=timestep_shift,
                        num_timesteps=num_timesteps_gen,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type_gen,
                        initial_noise=initial_noise[round_idx] if initial_noise is not None else None,
                        enable_sde=enable_sde,
                        generator=generator,
                        sde_timestep_idx=sde_timestep_idx,
                    )
                    output_dict[f'Round_{round_idx}/mean_cot_length'] = extra_info[f'Round_{round_idx}/mean_cot_length']
                    for i in range(batch_size):
                        current_images[i] = Image.fromarray(img[i])
                        output_dict['per_sample_results'][i].append(batch_rollout_output[i])
                        output_dict['per_sample_format_rewards'][i].append(extra_info[f'Round_{round_idx}/per_sample_format_rewards'][i])
                        output_dict['per_sample_imgs'][i].append(current_images[i])

                if round_idx > 0:
                    edit_instructions, batch_rollout_output, extra_info = self.get_edit_think_and_operation(
                        round_idx=round_idx,
                        step_idx=0,
                        input_imgs=current_images,
                        prompts=reflection_prompts,
                        max_output_token_n=max_output_token_n_edit,
                        do_sample=do_sample,
                        text_temperature=text_temperature,
                        topk=topk,
                        use_vae_feature=False,
                        generator=generator,
                    )
                    output_dict[f'Round_{round_idx}/mean_cot_length'] = extra_info[f'Round_{round_idx}/mean_cot_length']
                    for i in range(batch_size):
                        output_dict['per_sample_results'][i].append(batch_rollout_output[i])
                        output_dict['per_sample_format_rewards'][i].append(extra_info[f'Round_{round_idx}/per_sample_format_rewards'][i])

                    edit_operations = self.extract_edit_operation(edit_instructions)
                    output_dict['per_sample_edit_operations'] = edit_operations
                    for edit_idx, edit_operation in enumerate(edit_operations):
                        if "no further edit" in edit_operation.lower() and edit_idx in active_seq_idx and auto_stop:
                            output_dict['no_edit_cnt'][round_idx] += 1
                            active_seq_idx.remove(edit_idx)
                            output_dict['per_sample_finished_steps'][edit_idx] = round_idx
                            output_dict['per_sample_finished'][edit_idx] = True
                            last_future_reward = output_dict['per_sample_rewards'][edit_idx][-1]
                            output_dict['per_sample_rewards'][edit_idx].extend([last_future_reward] * (rounds - round_idx))
                            output_dict['per_sample_imgs'][edit_idx].extend([current_images[edit_idx]] * (rounds - round_idx))
                    # edit_operations = edit_instructions
                    img, batch_rollout_output, extra_info = self.edit_image(
                        round_idx=round_idx,
                        step_idx=1,
                        input_imgs=current_images,
                        edit_operations=edit_operations,
                        think=think,
                        max_output_token_n=max_output_token_n_edit,
                        do_sample=do_sample,
                        text_temperature=text_temperature,
                        topk=topk,
                        cfg_text_scale=cfg_text_scale,
                        cfg_img_scale=cfg_img_scale,
                        cfg_interval=cfg_interval,
                        timestep_shift=timestep_shift,
                        num_timesteps=num_timesteps_edit,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type_edit,
                        initial_noise=initial_noise[round_idx] if initial_noise is not None else None,
                        enable_sde=enable_sde,
                        sde_timestep_idx=sde_timestep_idx,
                    )

                    for i in range(batch_size):
                        current_images[i] = Image.fromarray(img[i])
                        output_dict['per_sample_results'][i].append(batch_rollout_output[i])


                for i in range(batch_size):
                    if i not in active_seq_idx and auto_stop:
                        continue
                    output_dict['per_sample_imgs'][i].append(current_images[i])
                    # get reward
                    if reward_fn_type == "geneval":
                        output_dict['per_sample_rewards'][i].append(
                            executor.submit(reward_fn, [img[i]], [prompts[i]], [prompt_metadata[i]])
                        )
                    elif reward_fn_type == "geneval_plus":
                        output_dict['per_sample_rewards'][i].append(
                            executor.submit(reward_fn, img[i], prompts[i], prompt_metadata[i])
                        )
                    else:
                        output_dict['per_sample_rewards'][i].append(
                            executor.submit(reward_fn, img[i], prompts[i], prompt_metadata[i])
                        )
            
        return output_dict
    

    @torch.no_grad()
    def rollout(
        self, 
        rounds: int = 1,
        prompts: List[str] = None,
        reflection_prompts: List[str] = None,
        start_round_idx: int = 0,
        max_output_token_n_gen: int = 256,
        max_output_token_n_edit: int = 512, 
        do_sample: bool = True,
        text_temperature: float = 0.3,
        think: bool = True,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps_gen=50,
        num_timesteps_edit=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type_gen="global",
        cfg_renorm_type_edit="global",
        image_shapes: Tuple[int, int] = (1024, 1024),
        initial_noise: Optional[List[torch.Tensor]] = None,
        topk: int = -1,
        executor: Optional[Executor] = None,
        reward_fn: Callable = None, # reward function for each round, return a list of rewards for each sample
        reward_fn_type: str = "unified_reward", # unified_reward or geneval
        prompt_metadata: List[Dict] = None, # metadata for each prompt
        enable_sde: bool = False,
        use_vae_feature: bool = False,
        generator: torch.Generator = None,
        input_imgs: List[Image.Image] = None,
        sde_timestep_idx: List[int] = None,
    ):
        assert rounds >= 1, "rounds must be >= 1"
        
        batch_size = len(prompts)
        image_shapes = [image_shapes] * batch_size
        
        # Initialize output dictionary with multi-round structure
        output_dict = {
            'rounds': rounds,
            # per_sample_results: List of results for each sample, each sample_i has a list of n_round_i results
            'per_sample_results': [[] for _ in range(batch_size)],
            # per_sample_rewards: List of rewards for each sample, each sample_i has a list of n_round_i rewards
            'per_sample_rewards': [[] for _ in range(batch_size)],
            # per_sample_format_rewards: List of format rewards for each sample, each sample_i has a list of n_round_i format rewards
            'per_sample_format_rewards': [[] for _ in range(batch_size)],
            'output_imgs': []
        }
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            current_images = input_imgs if input_imgs is not None else [None] * batch_size  # Track current images for each sample
            
            for round_idx in range(start_round_idx, rounds):
                if round_idx == 0:
                    img, batch_rollout_output, extra_info = self.generate_image_with_think(
                        round_idx=round_idx, 
                        prompts=prompts,
                        image_shapes=image_shapes,
                        think=think,
                        max_output_token_n=max_output_token_n_gen,
                        do_sample=do_sample,
                        text_temperature=text_temperature,
                        topk=topk,
                        cfg_text_scale=cfg_text_scale,
                        cfg_interval=cfg_interval,
                        timestep_shift=timestep_shift,
                        num_timesteps=num_timesteps_gen,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type_gen,
                        initial_noise=initial_noise[round_idx] if initial_noise is not None else None,
                        enable_sde=enable_sde,
                        generator=generator,
                        sde_timestep_idx=sde_timestep_idx,
                    )
                    output_dict[f'Round_{round_idx}/mean_cot_length'] = extra_info[f'Round_{round_idx}/mean_cot_length']
                    for i in range(batch_size):
                        current_images[i] = Image.fromarray(img[i])
                        output_dict['output_imgs'].append(current_images[i])
                        output_dict['per_sample_results'][i].append(batch_rollout_output[i])
                        output_dict['per_sample_format_rewards'][i].append(extra_info[f'Round_{round_idx}/per_sample_format_rewards'][i])


                if round_idx > 0:
                    actual_round_idx = round_idx if start_round_idx == 0 else 0
                    edit_instructions, batch_rollout_output, extra_info = self.get_edit_think_and_operation(
                        round_idx=actual_round_idx,
                        step_idx=0,
                        input_imgs=current_images,
                        prompts=reflection_prompts,
                        max_output_token_n=max_output_token_n_edit,
                        do_sample=do_sample,
                        text_temperature=text_temperature,
                        topk=topk,
                        use_vae_feature=False,
                        generator=generator,
                    )
                    output_dict[f'Round_{actual_round_idx}/mean_cot_length'] = extra_info[f'Round_{actual_round_idx}/mean_cot_length']
                    for i in range(batch_size):
                        output_dict['per_sample_format_rewards'][i].append(extra_info[f'Round_{actual_round_idx}/per_sample_format_rewards'][i])
                        output_dict['per_sample_results'][i].append(batch_rollout_output[i])

                    edit_operations = self.extract_edit_operation(edit_instructions)
                    output_dict['per_sample_edit_operations'] = edit_operations
                    # edit_operations = edit_instructions
                    img, batch_rollout_output, extra_info = self.edit_image(
                        round_idx=actual_round_idx,
                        step_idx=1,
                        input_imgs=current_images,
                        edit_operations=edit_operations,
                        think=think,
                        max_output_token_n=max_output_token_n_edit,
                        do_sample=do_sample,
                        text_temperature=text_temperature,
                        topk=topk,
                        cfg_text_scale=cfg_text_scale,
                        cfg_img_scale=cfg_img_scale,
                        cfg_interval=cfg_interval,
                        timestep_shift=timestep_shift,
                        num_timesteps=num_timesteps_edit,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type_edit,
                        initial_noise=initial_noise[round_idx],
                        enable_sde=enable_sde,
                        sde_timestep_idx=sde_timestep_idx,
                    )

                    for i in range(batch_size):
                        current_images[i] = Image.fromarray(img[i])
                        output_dict['output_imgs'].append(current_images[i])
                        output_dict['per_sample_results'][i].append(batch_rollout_output[i])


                for i in range(batch_size):
                    # get reward
                    if reward_fn_type == "geneval":
                        output_dict['per_sample_rewards'][i].append(
                            executor.submit(reward_fn, [img[i]], [prompts[i]], [prompt_metadata[i]])
                        )
                    elif reward_fn_type == "geneval_plus":
                        output_dict['per_sample_rewards'][i].append(
                            executor.submit(reward_fn, img[i], prompts[i], prompt_metadata[i])
                        )
                    elif reward_fn_type == "tiif":
                        output_dict['per_sample_rewards'][i].append(
                            executor.submit(reward_fn, img[i], prompts[i], prompt_metadata[i])
                        )
                    else:
                        raise NotImplementedError(f"Reward function type {reward_fn_type} not implemented")
            
        return output_dict
    

    def prepack(self, output_dict, with_text_loss=True, with_img_loss=True, 
                   timestep_ratio=1, pack_start_round_idx=0, pack_end_round_idx=1):
        """
        Prepack multi-round rollout results for training.
        
        Args:
            output_dict: Multi-round rollout results with new data structure
            with_text_loss: Whether to include text in loss computation
            with_img_loss: Whether to include image in loss computation
            timestep_ratio: Ratio of timesteps to sample for image loss
            pack_start_round_idx: Start round index to pack
            pack_end_round_idx: End round index to packs
        """
        # batch num = number valid rounds
        data_points_per_round = [[]]
        total_text_grad_tokens = 0
        total_image_num = 0
        
        batch_size = len(output_dict['per_sample_results'])
        
        # Process each sample's results across all rounds - one data point per sample per round
        for sample_idx in range(batch_size):
            # sample_results: List of results for each sample, [[StepResult, StepResult, ...], ...]
            sample_results = output_dict['per_sample_results'][sample_idx]
            for step_result_list in sample_results:
                step_result_list: List[RolloutStepResult] # contains 1-3 RolloutStepResult, 1 for conditional, 2 for cfg text, 3 for cfg image
                round_idx = step_result_list[0].round_idx
                step_idx = step_result_list[0].step_idx
                if round_idx < pack_start_round_idx or round_idx > pack_end_round_idx:
                    continue
                length_of_result_list = len(step_result_list)
                sample_has_loss = False
                for idx, (content_type, with_loss) in enumerate(zip(step_result_list[0].content_type_list, step_result_list[0].need_loss_list)):
                    if content_type == "text":
                        step_result_list[0].need_loss_list[idx] &= with_text_loss
                        sample_has_loss |= step_result_list[0].need_loss_list[idx]
                        if step_result_list[0].need_loss_list[idx]:
                            text_tokens = self.tokenizer.encode(step_result_list[0].output_list[idx])
                            total_text_grad_tokens += (len(text_tokens) - 1)
                    elif content_type == "image_and_latent":
                        step_result_list[0].need_loss_list[idx] &= with_img_loss
                        sample_has_loss |= step_result_list[0].need_loss_list[idx]
                        if step_result_list[0].need_loss_list[idx]:
                            num_timesteps = step_result_list[0].timestep.shape[0]
                            timestep_indices = list(range(num_timesteps))
                            sampled_num_timesteps = max(1, int(num_timesteps * timestep_ratio))
                            total_image_num += sampled_num_timesteps
                            selected_timestep_indices = random.sample(timestep_indices, sampled_num_timesteps)
                            # also select timestep for cfg context
                            for data_idx in range(length_of_result_list):
                                latent = []
                                prev_latent = []
                                log_prob = []
                                timestep = []
                                dt = []
                                for timestep_idx in selected_timestep_indices:
                                    latent.append(step_result_list[data_idx].latent[timestep_idx])
                                    prev_latent.append(step_result_list[data_idx].latent[timestep_idx + 1])
                                    log_prob.append(step_result_list[data_idx].log_prob[timestep_idx])
                                    timestep.append(step_result_list[data_idx].timestep[timestep_idx])
                                    dt.append(step_result_list[data_idx].dt[timestep_idx])
                                step_result_list[data_idx].latent = latent
                                step_result_list[data_idx].prev_latent = prev_latent
                                step_result_list[data_idx].log_prob = log_prob
                                step_result_list[data_idx].timestep = timestep
                                step_result_list[data_idx].dt = dt
                # if this sample has loss, add it to the data points
                if sample_has_loss:
                    # NOTE: we pack all data to one list for now
                    data_points_per_round[0].extend(step_result_list)

        return data_points_per_round, total_text_grad_tokens, total_image_num
    
    def pack_iterator(self, data_points_per_round, text_advantages, image_advantages, cfg_interval=(0.4, 1.0)):
        """
        Pack multi-round data points into batches for training.
        Similar to RolloutControllerForMaze.pack_iterator.

        text_advantages: List[List[float]], image_advantages: List[List[float]], [sample_idx][round_idx]
        """
        for data_points in data_points_per_round:
            if len(data_points):
                packed_batch, loss_info = self.pack_batch(data_points, 
                        text_advantages, image_advantages, "cuda",
                        cfg_interval=cfg_interval)
                yield packed_batch, loss_info

        
    def init_packed_input(self,):
        return {
            'sequence_length': 0,
            'sample_lens': [],
            'split_lens': [],
            'attn_modes': [],
            'packed_text_ids': [],
            'packed_text_indexes': [],
            'packed_position_ids': [],
            'ce_loss_indexes': [],
            'packed_noisy_latent': [],
            'patchified_vae_latent_shapes': [],
            'packed_latent_position_ids': [],
            'packed_vae_token_indexes': [],
            'packed_timesteps': [],
            'mse_loss_indexes': [],
            'packed_vit_tokens': [],
            'packed_vit_token_indexes': [],
            'packed_vit_position_ids': [],
            'vit_token_seqlens': [],
        }

    def init_loss_info(self,):
        return {
            'text_advantages': [],
            'image_advantages': [],
            'dts': [],
            'log_probs': [],
            'packed_label_ids': [],
            'ce_loss_weights': [],
            'packed_prev_latents': [],
            'packed_timesteps_for_loss': [],
            'packed_noisy_latent_for_loss': [],
            'patchified_vae_latent_shapes_for_loss': [],
            'mse_condition_token_indexes': [],
        }
    
    def add_text(self, text, packed_input, loss_info, with_loss=False, advantage=0, curr_pos=0, curr_rope_id=0):
        """
        Add text to packed input structure.
        
        Args:
            text: Text content to add
            packed_input: Packed input dictionary to update
            loss_info: Loss info dictionary to update  
            with_loss: Whether to include this text in loss computation
            advantage: Advantage value for this text
            curr_pos: Current sequence position
            curr_rope_id: Current rope position ID
            
        Returns:
            Updated curr_pos, curr_rope_id
        """
        # Encode text
        text_ids = self.tokenizer.encode(text)
        full_text_ids = [self.new_token_ids['bos_token_id']] + text_ids + [self.new_token_ids['eos_token_id']]
        full_text_len = len(full_text_ids)
        
        # Add to packed input
        packed_input['packed_text_ids'].extend(full_text_ids)
        packed_input['packed_text_indexes'].extend(range(curr_pos, curr_pos + full_text_len))
        packed_input['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + full_text_len))
        
        # Add loss information if needed
        if with_loss:
            packed_input['ce_loss_indexes'].extend(range(curr_pos, curr_pos + full_text_len - 1))
            loss_info['ce_loss_weights'].extend([1.0] * (full_text_len - 1))
            loss_info['packed_label_ids'].extend(full_text_ids[1:])
            loss_info['text_advantages'].extend([advantage] * (full_text_len - 1))
        
        # Update split info
        packed_input['split_lens'].append(full_text_len)
        packed_input['attn_modes'].append("causal")
        
        return curr_pos + full_text_len, curr_rope_id + full_text_len

    def add_image(self, image, packed_input, loss_info, curr_pos=0, curr_rope_id=0, need_vae=True, need_vit=True):
        """
        Add image (both VAE and VIT) to packed input structure.
        
        Args:
            image: PIL Image to add
            packed_input: Packed input dictionary to update
            loss_info: Loss info dictionary to update
            curr_pos: Current sequence position
            curr_rope_id: Current rope position ID
            
        Returns:
            Updated curr_pos, curr_rope_id
        """
        # Process VAE image
        if need_vae:
            curr_pos, curr_rope_id = self._add_vae_image(image, packed_input, curr_pos, curr_rope_id)
        
        # Process VIT image  
        if need_vit:
            curr_pos, curr_rope_id = self._add_vit_image(image, packed_input, curr_pos, curr_rope_id)
        
        return curr_pos, curr_rope_id
    
    def _add_vae_image(self, image, packed_input, curr_pos, curr_rope_id):
        """Add VAE processed image to packed input."""
        # Add start of image token
        packed_input['packed_text_ids'].append(self.new_token_ids['start_of_image'])
        packed_input['packed_text_indexes'].append(curr_pos)
        curr_pos += 1
        curr_split_len = 1
        
        # Process VAE image
        vae_image_tensor = self.vae_transform(image)
        packed_input['_padded_images'].append(vae_image_tensor)  # Temporary storage
        
        H, W = vae_image_tensor.shape[1:]
        vae_downsample_factor = self.vae_transform.stride
        h_vae, w_vae = H // vae_downsample_factor, W // vae_downsample_factor
        packed_input['patchified_vae_latent_shapes'].append((h_vae, w_vae))
        
        # Get position IDs for VAE latent
        vae_latent_position_ids = get_flattened_position_ids_extrapolate(
            H, W, vae_downsample_factor, 
            max_num_patches_per_side=self.unwrapped_model.config.max_latent_size
        )
        packed_input['packed_latent_position_ids'].append(vae_latent_position_ids)
        
        num_vae_tokens = h_vae * w_vae
        packed_input['packed_vae_token_indexes'].extend(range(curr_pos, curr_pos + num_vae_tokens))
        packed_input['packed_timesteps'].extend([0] * num_vae_tokens)  # No timestep for context images
        packed_input['_vae_latent_plan'].append("clean")  # Track latent type
        curr_pos += num_vae_tokens
        curr_split_len += num_vae_tokens
        
        # Add end of image token
        packed_input['packed_text_ids'].append(self.new_token_ids['end_of_image'])
        packed_input['packed_text_indexes'].append(curr_pos)
        curr_pos += 1
        curr_split_len += 1
        
        # Update sequence status
        packed_input['attn_modes'].append("full")
        packed_input['split_lens'].append(curr_split_len)
        packed_input['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
        
        return curr_pos, curr_rope_id + 1
    
    def _add_vit_image(self, image, packed_input, curr_pos, curr_rope_id):
        """Add VIT processed image to packed input."""
        # Add start of image token for VIT
        packed_input['packed_text_ids'].append(self.new_token_ids['start_of_image'])
        packed_input['packed_text_indexes'].append(curr_pos)
        curr_pos += 1
        curr_split_len = 1
        
        # Process VIT image
        vit_image_tensor = self.vit_transform(image)
        H_vit, W_vit = vit_image_tensor.shape[1:]
        vit_downsample_factor = self.vit_transform.stride
        h_vit, w_vit = H_vit // vit_downsample_factor, W_vit // vit_downsample_factor
        vit_image_tensor = patchify(vit_image_tensor, vit_downsample_factor)
        
        packed_input['vit_token_seqlens'].append(vit_image_tensor.shape[0])
        packed_input['packed_vit_tokens'].append(vit_image_tensor)
        
        # Get position IDs for VIT latent
        vit_latent_position_ids = get_flattened_position_ids_extrapolate(
            H_vit, W_vit, vit_downsample_factor, 
            max_num_patches_per_side=self.unwrapped_model.config.vit_max_num_patch_per_side
        )
        packed_input['packed_vit_position_ids'].append(vit_latent_position_ids)
        
        num_vit_tokens = h_vit * w_vit
        packed_input['packed_vit_token_indexes'].extend(range(curr_pos, curr_pos + num_vit_tokens))
        curr_pos += num_vit_tokens
        curr_split_len += num_vit_tokens
        
        # Add end of image token
        packed_input['packed_text_ids'].append(self.new_token_ids['end_of_image'])
        packed_input['packed_text_indexes'].append(curr_pos)
        curr_pos += 1
        curr_split_len += 1
        
        # Update sequence status
        packed_input['attn_modes'].append("full")
        packed_input['split_lens'].append(curr_split_len)
        packed_input['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
        
        return curr_pos, curr_rope_id + 1

    def add_latent(self, latent_list, dt_list, log_prob_list, timestep_list, prev_latent_list, 
                   image_size, packed_input, loss_info, with_loss=False, advantage=0, 
                   curr_pos=0, curr_rope_id=0):
        """
        Add noisy latents to packed input structure.
        
        Args:
            latent_list: List of noisy latents 
            dt_list: List of dt values
            log_prob_list: List of log probabilities
            timestep_list: List of timesteps
            prev_latent_list: List of previous latents
            image_size: Image size tuple (H, W)
            packed_input: Packed input dictionary to update
            loss_info: Loss info dictionary to update
            with_loss: Whether to include in loss computation
            advantage: Advantage value
            curr_pos: Current sequence position
            curr_rope_id: Current rope position ID
            
        Returns:
            Updated curr_pos, curr_rope_id
        """
        stage_rope_id = curr_rope_id
        
        for i, (noisy_latent, dt, log_prob, timestep, prev_latent) in enumerate(
            zip(latent_list, dt_list, log_prob_list, timestep_list, prev_latent_list)):
            
            # Add start of image token
            packed_input['packed_text_ids'].append(self.new_token_ids['start_of_image'])
            packed_input['packed_text_indexes'].append(curr_pos)
            curr_pos += 1
            curr_split_len = 1
            
            # Add noisy latent
            packed_input['packed_noisy_latent'].append(noisy_latent)
            packed_input['_vae_latent_plan'].append("noise")
            
            # Calculate dimensions from latent shape
            H, W = image_size
            downsample_factor = self.vae_transform.stride
            h, w = H // downsample_factor, W // downsample_factor
            packed_input['patchified_vae_latent_shapes'].append((h, w))
            
            # Get position IDs for latent tokens
            latent_position_ids = get_flattened_position_ids_extrapolate(
                H, W, downsample_factor, 
                max_num_patches_per_side=self.unwrapped_model.config.max_latent_size
            )
            packed_input['packed_latent_position_ids'].append(latent_position_ids)
            
            num_latent_tokens = noisy_latent.shape[0]
            packed_input['packed_vae_token_indexes'].extend(range(curr_pos, curr_pos + num_latent_tokens))
            packed_input['mse_loss_indexes'].extend(range(curr_pos, curr_pos + num_latent_tokens))
            if with_loss:
                # For loss computation
                packed_input['_patchified_vae_latent_shapes_for_loss'].append((h, w))
                packed_input['_packed_noisy_latent_for_loss'].append(noisy_latent)
                
                loss_info['dts'].extend([dt] * num_latent_tokens)
                loss_info['log_probs'].append(log_prob)
                loss_info['packed_prev_latents'].append(prev_latent)
                loss_info['packed_timesteps_for_loss'].extend([timestep] * num_latent_tokens)
                loss_info['image_advantages'].append(advantage)
            
            packed_input['packed_timesteps'].extend([timestep] * num_latent_tokens)
            curr_pos += num_latent_tokens
            curr_split_len += num_latent_tokens
            
            # Add end of image token
            packed_input['packed_text_ids'].append(self.new_token_ids['end_of_image'])
            packed_input['packed_text_indexes'].append(curr_pos)
            curr_pos += 1
            curr_split_len += 1
            
            # Update sequence status
            packed_input['attn_modes'].append("noise")
            packed_input['split_lens'].append(curr_split_len)
            packed_input['packed_position_ids'].extend([stage_rope_id] * curr_split_len)
        
        return curr_pos, curr_rope_id

    def pack_batch(self, current_batch: List[RolloutStepResult], text_advantages=None, image_advantages=None, device="cuda", cfg_interval=(0.4, 1.0)):
        """
        Pack a batch of RolloutStepResult into format suitable for Bagel.forward().
        Uses modular add_text, add_image, add_latent functions for cleaner code.
        Handles CFG data points based on is_cfg_text and is_cfg_img flags.
        """
        packed_input = self.init_packed_input()
        loss_info = self.init_loss_info()
        
        # Add temporary fields for processing
        packed_input['_padded_images'] = []
        packed_input['_vae_latent_plan'] = []
        packed_input['_packed_noisy_latent_for_loss'] = []
        packed_input['_patchified_vae_latent_shapes_for_loss'] = []
        
        curr_pos = 0
        noisy_latent_cnt = 0 # for logging how many conditional latents have been added
        noisy_latent_idx = 0 # for logging the index of the noisy latent in noisy latent sequence
        
        # Group data points by (sample_idx, round_idx) for CFG pairing
        grouped_data = {}
        for data_point in current_batch:
            key = (data_point.sample_idx, data_point.round_idx)
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(data_point)
        
        # Process all data points
        for (sample_idx, round_idx), data_points in grouped_data.items():    
            num_noisy_latent = 0                    
            for data_point in data_points:
                data_point: RolloutStepResult
                curr_rope_id = 0
                sample_start = curr_pos

                # Get advantage values (only for conditional data)
                if not (data_point.is_cfg_text or data_point.is_cfg_img):
                    text_adv = text_advantages[sample_idx][round_idx] if text_advantages else 0
                    image_adv = image_advantages[sample_idx][round_idx] if image_advantages else 0
                else:
                    text_adv = image_adv = 0
                
                for i, (output, content_type, need_loss) in enumerate(
                    zip(data_point.output_list, data_point.content_type_list, data_point.need_loss_list)):
                    
                    if content_type == 'text':
                        # CFG text doesn't contribute to loss
                        with_loss = need_loss and not (data_point.is_cfg_text or data_point.is_cfg_img)
                        curr_pos, curr_rope_id = self.add_text(
                            text=output,
                            packed_input=packed_input,
                            loss_info=loss_info,
                            with_loss=with_loss,
                            advantage=text_adv,
                            curr_pos=curr_pos,
                            curr_rope_id=curr_rope_id
                        )
                        
                    elif content_type == 'image':
                        curr_pos, curr_rope_id = self.add_image(
                            image=output,
                            packed_input=packed_input,
                            loss_info=loss_info,
                            curr_pos=curr_pos,
                            curr_rope_id=curr_rope_id,
                            need_vae=data_point.need_vae,
                            need_vit=data_point.need_vit
                        )
                        
                    elif content_type == 'image_and_latent' and data_point.latent is not None and need_loss:
                        # Handle latent processing differently for conditional vs CFG
                        if data_point.is_cfg_text or data_point.is_cfg_img:
                            # CFG latent processing
                            num_noisy_latent = len(data_point.timestep)
                            num_tokens_per_latent = data_point.latent[0].shape[0]
                            filtered_indices = [j for j, timestep in enumerate(data_point.timestep) 
                                              if cfg_interval[0] <= timestep <= cfg_interval[1]]
                            
                            if filtered_indices:
                                curr_pos, curr_rope_id = self.add_latent(
                                    latent_list=[data_point.latent[j] for j in filtered_indices],
                                    dt_list=[data_point.dt[j] for j in filtered_indices],
                                    log_prob_list=[data_point.log_prob[j] for j in filtered_indices],
                                    timestep_list=[data_point.timestep[j] for j in filtered_indices],
                                    prev_latent_list=[data_point.prev_latent[j] for j in filtered_indices],
                                    image_size=output.size,
                                    packed_input=packed_input,
                                    loss_info=loss_info,
                                    with_loss=False,  # CFG doesn't contribute to loss info
                                    advantage=0,
                                    curr_pos=curr_pos,
                                    curr_rope_id=curr_rope_id
                                )
                                for filtered_idx in filtered_indices:
                                    loss_info['mse_condition_token_indexes'][noisy_latent_cnt+filtered_idx].append(
                                        (noisy_latent_idx, noisy_latent_idx + num_tokens_per_latent)
                                    )
                                    noisy_latent_idx += num_tokens_per_latent
                        else:
                            # Conditional latent processing
                            curr_pos, curr_rope_id = self.add_latent(
                                latent_list=data_point.latent,
                                dt_list=data_point.dt,
                                log_prob_list=data_point.log_prob,
                                timestep_list=data_point.timestep,
                                prev_latent_list=data_point.prev_latent,
                                image_size=output.size[::-1],
                                packed_input=packed_input,
                                loss_info=loss_info,
                                with_loss=True,
                                advantage=image_adv,
                                curr_pos=curr_pos,
                                curr_rope_id=curr_rope_id
                            )
                            num_noisy_latent = len(data_point.timestep)
                            num_tokens_per_latent = data_point.latent[0].shape[0]
                            for j in range(num_noisy_latent):
                                loss_info['mse_condition_token_indexes'].append([(noisy_latent_idx, noisy_latent_idx + num_tokens_per_latent)])
                                noisy_latent_idx += num_tokens_per_latent


                sample_len = curr_pos - sample_start
                packed_input['sample_lens'].append(sample_len)
            noisy_latent_cnt += num_noisy_latent
        
        # Post-process and convert to tensors
        packed_input['sequence_length'] = curr_pos
        
        # Handle padding
        pad_len = 65536 - curr_pos
        if pad_len > 0:
            packed_input['split_lens'].append(pad_len)
            packed_input['attn_modes'].append('causal')
            packed_input['sample_lens'].append(pad_len)
        elif pad_len < 0:
            raise ValueError(f"Sequence too long: {curr_pos} > 65536")
        
        # Convert lists to tensors
        self._convert_to_tensors(packed_input, loss_info, device)
        
        # Process VAE images if any
        self._process_vae_images(packed_input, device)

        # Clean up temporary fields
        del packed_input['_padded_images']
        del packed_input['_vae_latent_plan']
        if '_packed_noisy_latent_for_loss' in packed_input:
            del packed_input['_packed_noisy_latent_for_loss']
        if '_patchified_vae_latent_shapes_for_loss' in packed_input:
            del packed_input['_patchified_vae_latent_shapes_for_loss']
        
        return packed_input, loss_info
    

    def _convert_to_tensors(self, packed_input, loss_info, device):
        """Convert lists in packed_input and loss_info to tensors."""
        # text related
        packed_input['packed_text_ids'] = torch.tensor(packed_input['packed_text_ids'], dtype=torch.long, device=device)
        packed_input['packed_text_indexes'] = torch.tensor(packed_input['packed_text_indexes'], dtype=torch.long, device=device)
        packed_input['packed_position_ids'] = torch.tensor(packed_input['packed_position_ids'], dtype=torch.long, device=device)

        if packed_input['ce_loss_indexes']: packed_input['ce_loss_indexes'] = torch.tensor(packed_input['ce_loss_indexes'], dtype=torch.long, device=device)
        else: packed_input['ce_loss_indexes'] = None

        # VAE tensors
        if packed_input['packed_vae_token_indexes']: packed_input['packed_vae_token_indexes'] = torch.tensor(packed_input['packed_vae_token_indexes'], dtype=torch.long, device=device)
        else: packed_input['packed_vae_token_indexes'] = None

        if packed_input['mse_loss_indexes']: packed_input['mse_loss_indexes'] = torch.tensor(packed_input['mse_loss_indexes'], dtype=torch.long, device=device)
        else: packed_input['mse_loss_indexes'] = None

        if packed_input['packed_timesteps']: packed_input['packed_timesteps'] = torch.tensor(packed_input['packed_timesteps'], dtype=torch.float32, device=device)
        else: packed_input['packed_timesteps'] = None

        if packed_input['patchified_vae_latent_shapes']: packed_input['patchified_vae_latent_shapes'] = torch.tensor(packed_input['patchified_vae_latent_shapes'], dtype=torch.long, device=device)
        else: packed_input['patchified_vae_latent_shapes'] = None

        if packed_input['packed_latent_position_ids']: packed_input['packed_latent_position_ids'] = torch.cat(packed_input['packed_latent_position_ids']).to(device)
        else: packed_input['packed_latent_position_ids'] = None
        # VIT tensors
        if packed_input['packed_vit_tokens']: packed_input['packed_vit_tokens'] = torch.cat(packed_input['packed_vit_tokens']).to(device)
        else: packed_input['packed_vit_tokens'] = None

        if packed_input['packed_vit_position_ids']: packed_input['packed_vit_position_ids'] = torch.cat(packed_input['packed_vit_position_ids']).to(device)
        else: packed_input['packed_vit_position_ids'] = None

        if packed_input['packed_vit_token_indexes']: packed_input['packed_vit_token_indexes'] = torch.tensor(packed_input['packed_vit_token_indexes'], dtype=torch.long, device=device)
        else: packed_input['packed_vit_token_indexes'] = None

        if packed_input['vit_token_seqlens']: packed_input['vit_token_seqlens'] = torch.tensor(packed_input['vit_token_seqlens'], dtype=torch.long, device=device)
        else: packed_input['vit_token_seqlens'] = None
                    
        # Handle packed_noisy_latent - will be processed later in _process_vae_images
        # For now, just ensure it's a list or None
        if not packed_input.get('packed_noisy_latent'): packed_input['packed_noisy_latent'] = None
        
        # Convert loss_info lists to tensors  
        if loss_info['text_advantages']: loss_info['text_advantages'] = torch.tensor(loss_info['text_advantages'], dtype=torch.float32, device=device)
        else: loss_info['text_advantages'] = None
            
        if loss_info['image_advantages']: loss_info['image_advantages'] = torch.tensor(loss_info['image_advantages'], dtype=torch.float32, device=device)
        else: loss_info['image_advantages'] = None
            
        if loss_info['dts']: loss_info['dts'] = torch.tensor(loss_info['dts'], dtype=torch.float32, device=device)
        else: loss_info['dts'] = None
            
        if loss_info['ce_loss_weights']: loss_info['ce_loss_weights'] = torch.tensor(loss_info['ce_loss_weights'], dtype=torch.float32, device=device)
        else: loss_info['ce_loss_weights'] = None
            
        if loss_info['packed_label_ids']: loss_info['packed_label_ids'] = torch.tensor(loss_info['packed_label_ids'], dtype=torch.long, device=device)
        else: loss_info['packed_label_ids'] = None
            
        if loss_info['packed_timesteps_for_loss']: loss_info['packed_timesteps_for_loss'] = torch.tensor(loss_info['packed_timesteps_for_loss'], dtype=torch.float32, device=device)
        else: loss_info['packed_timesteps_for_loss'] = None
            
        if packed_input.get('_patchified_vae_latent_shapes_for_loss'): loss_info['patchified_vae_latent_shapes_for_loss'] = torch.tensor(packed_input['_patchified_vae_latent_shapes_for_loss'], dtype=torch.long, device=device)
        else: loss_info['patchified_vae_latent_shapes_for_loss'] = None
            
        # Concatenate tensor lists in loss_info
        if loss_info['log_probs']: loss_info['log_probs'] = torch.cat(loss_info['log_probs'], dim=0).to(device)
        else: loss_info['log_probs'] = None
            
        if loss_info['packed_prev_latents']: loss_info['packed_prev_latents'] = torch.cat(loss_info['packed_prev_latents'], dim=0).to(device)
        else: loss_info['packed_prev_latents'] = None
            
        if packed_input.get('_packed_noisy_latent_for_loss'): loss_info['packed_noisy_latent_for_loss'] = torch.cat(packed_input['_packed_noisy_latent_for_loss'], dim=0).to(device)
        else: loss_info['packed_noisy_latent_for_loss'] = None
    
    def _process_vae_images(self, packed_input, device):
        """Process and encode VAE images."""
        
        if not packed_input.get('_padded_images'):
            # No clean images to process, just handle noisy latents
            if packed_input['packed_noisy_latent']:
                packed_input['packed_noisy_latent'] = torch.cat(packed_input['packed_noisy_latent'], dim=0).to(device)
            return
            
        padded_images = torch.stack(packed_input['_padded_images'], dim=0).to(device)
        
        with torch.inference_mode():
            p = self.vae_transform.stride // 8
            packed_latent = self.vae_model.encode(padded_images)
            b, c, h, w = packed_latent.shape
            packed_latent = packed_latent.reshape(b, c, h//p, p, w//p, p)
            packed_latent = torch.einsum("bchpwq->bhwpqc", packed_latent)
            packed_latent = packed_latent.reshape(b, h*w//p//p, c*p*p)
        
        # Rearrange noisy latents to include encoded clean images
        if packed_input.get('packed_noisy_latent'):
            cur_clean_idx = 0
            cur_noise_idx = 0
            rearranged_latents = []
            
            for latent_type in packed_input['_vae_latent_plan']:
                if latent_type == "clean":
                    rearranged_latents.append(packed_latent[cur_clean_idx].to(device))
                    cur_clean_idx += 1
                elif latent_type == "noise":
                    rearranged_latents.append(packed_input['packed_noisy_latent'][cur_noise_idx].to(device))
                    cur_noise_idx += 1
            
            if rearranged_latents:
                packed_input['packed_noisy_latent'] = torch.cat(rearranged_latents, dim=0)
            else:
                packed_input['packed_noisy_latent'] = None
        else:
            # Only clean images
            packed_input['packed_noisy_latent'] = packed_latent.reshape(-1, packed_latent.shape[-1])