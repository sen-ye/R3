# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from contextlib import contextmanager
from functools import partial
from typing import Optional, Set, Type, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard

from modeling.bagel.modeling_utils import MLPconnector, PositionEmbedding, TimestepEmbedder
from modeling.bagel.qwen2_navit import Qwen2DecoderLayer, Qwen2MoEDecoderLayer, Qwen2MoTDecoderLayer
from modeling.bagel.siglip_navit import SiglipEncoderLayer

logger = logging.getLogger(__name__)

DEFAULT_TRANSFORMER_LAYERS: Set[Type[nn.Module]] = {
    Qwen2DecoderLayer,
    Qwen2MoEDecoderLayer,
    Qwen2MoTDecoderLayer,
    SiglipEncoderLayer,
    MLPconnector,
    TimestepEmbedder,
    PositionEmbedding,
}


class FSDP2Config:
    def __init__(
        self,
        sharding_strategy: str = "HYBRID_SHARD",
        cpu_offload: bool = False,
        num_replicate: int = 1,
        num_shard: int = 8,
        mp_policy: str = "bf16",
        reshard_after_forward: Union[bool, str] = False,
    ):
        self.sharding_strategy = sharding_strategy
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard
        self.mp_policy = mp_policy
        self.reshard_after_forward = reshard_after_forward
        self._mesh_cache: Optional[DeviceMesh] = None

    def normalized_sharding_strategy(self) -> str:
        if self.sharding_strategy == "SHARD_GRAD_OP":
            return "FULL_SHARD"
        return self.sharding_strategy

    def resolve_num_replicate(self, world_size: Optional[int] = None) -> int:
        if world_size is None:
            world_size = dist.get_world_size()
        if self.num_shard <= 0:
            raise ValueError(f"num_shard must be positive, got {self.num_shard}")

        normalized_strategy = self.normalized_sharding_strategy()
        if normalized_strategy == "HYBRID_SHARD":
            if world_size % self.num_shard != 0:
                raise ValueError(
                    f"World size {world_size} must be divisible by num_shard {self.num_shard}"
                )
            self.num_replicate = world_size // self.num_shard
        else:
            self.num_replicate = max(1, world_size // self.num_shard)
        return self.num_replicate

    def get_device_mesh(self) -> DeviceMesh:
        if self._mesh_cache is not None:
            return self._mesh_cache

        world_size = dist.get_world_size()
        normalized_strategy = self.normalized_sharding_strategy()
        if normalized_strategy == "HYBRID_SHARD":
            num_replicate = self.resolve_num_replicate(world_size)
            if num_replicate * self.num_shard != world_size:
                raise ValueError(
                    f"Invalid FSDP2 mesh: replicate({num_replicate}) * shard({self.num_shard}) != world_size({world_size})"
                )
            self._mesh_cache = init_device_mesh(
                "cuda",
                mesh_shape=(num_replicate, self.num_shard),
                mesh_dim_names=("replicate", "shard"),
            )
        elif normalized_strategy == "FULL_SHARD":
            self.resolve_num_replicate(world_size)
            self._mesh_cache = init_device_mesh(
                "cuda",
                mesh_shape=(world_size,),
                mesh_dim_names=("shard",),
            )
        else:
            raise ValueError(f"Unknown FSDP2 sharding strategy: {self.sharding_strategy}")
        return self._mesh_cache

    def get_mixed_precision_policy(self) -> MixedPrecisionPolicy:
        if self.mp_policy == "bf16":
            return MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        if self.mp_policy == "fp16":
            return MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float16)
        return MixedPrecisionPolicy()

    def get_cpu_offload_policy(self):
        return CPUOffloadPolicy() if self.cpu_offload else None

    def get_reshard_after_forward(self) -> bool:
        if self.sharding_strategy == "SHARD_GRAD_OP":
            return False
        if self.reshard_after_forward == "default":
            return False
        return bool(self.reshard_after_forward)


def apply_fsdp2_with_activation_checkpointing(
    model: nn.Module,
    fsdp_config: FSDP2Config,
    transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None,
    use_ac: bool = True,
) -> nn.Module:
    if transformer_layer_cls is None:
        transformer_layer_cls = DEFAULT_TRANSFORMER_LAYERS

    if use_ac:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda submodule: isinstance(submodule, tuple(transformer_layer_cls)),
        )

    device_mesh = fsdp_config.get_device_mesh()
    mp_policy = fsdp_config.get_mixed_precision_policy()
    offload_policy = fsdp_config.get_cpu_offload_policy()
    reshard_after_forward = fsdp_config.get_reshard_after_forward()

    for module in model.modules():
        if isinstance(module, tuple(transformer_layer_cls)):
            fully_shard(
                module,
                mesh=device_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                reshard_after_forward=reshard_after_forward,
            )

    fully_shard(
        model,
        mesh=device_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=reshard_after_forward,
    )
    return model


class FSDP2Checkpoint:
    @staticmethod
    def save_checkpoint(
        ckpt_dir: str,
        train_steps: int,
        model: nn.Module,
        ema_model: Optional[nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler,
        data_status,
        logger,
        fsdp_config: FSDP2Config,
        save_safetensors: bool = True,
    ):
        save_path = os.path.join(ckpt_dir, f"ckpt-{train_steps:06d}")
        os.makedirs(save_path, exist_ok=True)

        dcp_state = {
            "model": get_model_state_dict(model),
            "optimizer": get_optimizer_state_dict(model, optimizer),
        }
        if ema_model is not None:
            dcp_state["ema"] = get_model_state_dict(ema_model)
        dcp.save(dcp_state, checkpoint_id=save_path)

        if dist.get_rank() == 0:
            if scheduler is not None:
                torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
            if data_status is not None:
                torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        if save_safetensors:
            try:
                model_state = get_model_state_dict(
                    model,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
                if dist.get_rank() == 0:
                    save_file(model_state, os.path.join(save_path, "model.safetensors"))
            except Exception as exc:
                logger.warning(f"FSDP2 safetensors export failed, DCP checkpoint is still saved: {exc}")

        dist.barrier()

    @staticmethod
    def load_model_checkpoint(
        resume_from: str,
        logger,
        model: nn.Module,
        ema_model: Optional[nn.Module] = None,
        load_ema_as_model: bool = False,
    ):
        if resume_from is None or not os.path.exists(resume_from):
            logger.info("No checkpoint found, training from scratch.")
            return model, ema_model

        if os.path.isdir(resume_from) and os.path.exists(os.path.join(resume_from, ".metadata")):
            state_dict_to_load = {"model": get_model_state_dict(model)}
            if ema_model is not None:
                state_dict_to_load["ema"] = get_model_state_dict(ema_model)
            dcp.load(state_dict_to_load, checkpoint_id=resume_from)
            set_model_state_dict(model, state_dict_to_load["model"])
            if ema_model is not None and "ema" in state_dict_to_load:
                set_model_state_dict(ema_model, state_dict_to_load["ema"])
            return model, ema_model

        safetensors_file = os.path.join(resume_from, "ema.safetensors" if load_ema_as_model else "model.safetensors")
        if not os.path.exists(safetensors_file):
            raise ValueError(f"No valid FSDP2 checkpoint found at {resume_from}")

        state_dict = load_file(safetensors_file, device="cpu")
        state_dict.pop("latent_pos_embed.pos_embed", None)
        state_dict.pop("vit_pos_embed.pos_embed", None)
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        if ema_model is not None and not load_ema_as_model:
            ema_file = os.path.join(resume_from, "ema.safetensors")
            if os.path.exists(ema_file):
                ema_state = load_file(ema_file, device="cpu")
                ema_state.pop("latent_pos_embed.pos_embed", None)
                ema_state.pop("vit_pos_embed.pos_embed", None)
                ema_model.load_state_dict(ema_state, strict=False)
        return model, ema_model

    @staticmethod
    def load_training_state(
        resume_from: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        fsdp_config: FSDP2Config,
    ):
        if resume_from is None or not os.path.exists(resume_from):
            return optimizer, scheduler, 0, None

        if os.path.isdir(resume_from):
            try:
                opt_state = get_optimizer_state_dict(model, optimizer)
                dcp.load({"optimizer": opt_state}, checkpoint_id=resume_from)
                set_optimizer_state_dict(model, optimizer, opt_state)
            except Exception as exc:
                logger.warning(f"Could not load FSDP2 optimizer state from DCP: {exc}")

        try:
            train_steps = int(os.path.basename(os.path.normpath(resume_from)).split("-")[-1])
        except ValueError:
            train_steps = 0

        scheduler_path = os.path.join(resume_from, "scheduler.pt")
        if os.path.exists(scheduler_path):
            scheduler_state = torch.load(scheduler_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state)

        data_status = None
        data_status_path = os.path.join(resume_from, "data_status.pt")
        if os.path.exists(data_status_path):
            data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")

        return optimizer, scheduler, train_steps, data_status


@torch.no_grad()
def fsdp2_ema_update(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999):
    if ema_model is None:
        return
    ema_params = [p for p in ema_model.parameters()]
    model_params = [p for p in model.parameters()]
    ema_target_params = []
    model_trainable_params = []
    for model_param, ema_param in zip(model_params, ema_params):
        if model_param.requires_grad:
            model_trainable_params.append(model_param)
            ema_target_params.append(ema_param)
    if model_trainable_params:
        torch._foreach_mul_(ema_target_params, decay)
        torch._foreach_add_(ema_target_params, model_trainable_params, alpha=1 - decay)


@torch.no_grad()
def clip_grad_norm_fsdp2(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    total_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
        foreach=foreach,
    )
    if hasattr(total_norm, "full_tensor"):
        total_norm = total_norm.full_tensor()
    return total_norm


@contextmanager
def fsdp2_rollout_context(model: nn.Module):
    fsdp_modules = [module for module in model.modules() if isinstance(module, FSDPModule)]
    was_training = model.training
    prev_grad_enabled = torch.is_grad_enabled()
    saved_hooks = {}
    saved_ckpt_forwards = {}
    unshard_handles = []

    try:
        model.eval()
        torch.set_grad_enabled(False)

        for module in fsdp_modules:
            handle = module.unshard(async_op=True)
            if handle is not None:
                unshard_handles.append(handle)
        for handle in unshard_handles:
            handle.wait()

        for module in fsdp_modules:
            saved = {}
            for attr in (
                "_forward_pre_hooks",
                "_forward_hooks",
                "_forward_pre_hooks_with_kwargs",
                "_forward_hooks_with_kwargs",
            ):
                hook_dict = getattr(module, attr, None)
                if hook_dict is not None and len(hook_dict) > 0:
                    saved[attr] = dict(hook_dict)
                    hook_dict.clear()
            saved_hooks[id(module)] = saved

        for module in model.modules():
            if isinstance(module, CheckpointWrapper):
                saved_ckpt_forwards[id(module)] = module.forward
                module.forward = module._checkpoint_wrapped_module.forward

        yield

    finally:
        for module in model.modules():
            if isinstance(module, CheckpointWrapper) and id(module) in saved_ckpt_forwards:
                module.forward = saved_ckpt_forwards[id(module)]

        for module in fsdp_modules:
            for attr, hook_dict in saved_hooks.get(id(module), {}).items():
                getattr(module, attr).update(hook_dict)

        for module in reversed(fsdp_modules):
            module.reshard()

        torch.cuda.empty_cache()
        torch.set_grad_enabled(prev_grad_enabled)
        model.train(was_training)


def fsdp2_lazy_init_root(model: nn.Module):
    if not isinstance(model, FSDPModule):
        return
    state = model._get_fsdp_state()
    state._lazy_init()
