import json
import os
import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union, cast

import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from safetensors.torch import load_file
from torch import nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers import T5EncoderModel, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Block

import genmo.mochi_preview.dit.joint_model.context_parallel as cp
import genmo.mochi_preview.vae.cp_conv as cp_conv
from genmo.lib.progress import get_new_progress_bar, progress_bar
from genmo.lib.utils import Timer
from genmo.mochi_preview.vae.models import (
    Decoder,
    decode_latents,
    decode_latents_tiled_full,
    decode_latents_tiled_spatial,
)
from genmo.mochi_preview.vae.vae_stats import dit_latents_to_vae_latents


def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule

#--------------------RESTART---------------------#
def generate_restart_t_steps(max_idx, restart_idx, num_steps, local_rank):
    """
    Generate restart_t_steps with integer spacing between max_idx and restart_idx.
    If exact spacing is not possible, approximate to closest integers.

    Args:
        max_idx (int): Starting index (inclusive).
        restart_idx (int): Ending index (inclusive).
        num_steps (int): Desired number of steps (-1 for all steps).

    Returns:
        torch.Tensor: Adjusted restart_t_steps.
    """
    if num_steps == -1:
        # Include all steps from max_idx to restart_idx
        restart_t_steps = torch.arange(max_idx, restart_idx)
    else:
        if num_steps <= 0:
            raise ValueError("num_steps must be greater than 0 or -1 for all steps")

        # Generate approximate evenly spaced values
        steps = torch.linspace(max_idx, restart_idx, steps=num_steps + 1)
        # Round to nearest integers
        restart_t_steps = torch.unique(steps.round().to(torch.int64))  # Remove duplicates caused by rounding

        # Ensure restart_t_steps contains both max_idx and restart_idx
        if restart_t_steps[0] != max_idx:
            restart_t_steps = torch.cat((torch.tensor([max_idx]), restart_t_steps))
        if restart_t_steps[-1] != restart_idx:
            restart_t_steps = torch.cat((restart_t_steps, torch.tensor([restart_idx])))

        restart_t_steps = torch.unique(restart_t_steps)  # Ensure uniqueness

    print(f"Generated restart_t_steps: {restart_t_steps.tolist()}") if local_rank == 0 else None
    return restart_t_steps

def add_noise_interval(x_t1, sigma_schedule, t1, t2):
    """
    Forward step from x_{t1} to x_{t2} using Gaussian noise.

    Args:
        x_t1: Sample at timestep t1.
        sigma_schedule: Pre-computed sigma values for all timesteps.
        t1: Current timestep.
        t2: Next timestep.
    
    Returns:
        x_t2: Sample at timestep t2.
    """
    sigma_t1 = sigma_schedule[t1]
    sigma_t2 = sigma_schedule[t2]

    rem_sigma_t1 = 1-sigma_t1
    rem_sigma_t2 = 1-sigma_t2

    sigma_ratio = rem_sigma_t2 / rem_sigma_t1

    epsilon = torch.randn_like(x_t1, dtype=x_t1.dtype) 

    x_t2 = sigma_ratio * x_t1 + (1 - sigma_ratio) * epsilon
    return x_t2
#--------------------------------------------------#

T5_MODEL = "google/t5-v1_1-xxl"
MAX_T5_TOKEN_LENGTH = 256


def setup_fsdp_sync(model, device_id, *, param_dtype, auto_wrap_policy) -> FSDP:
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=device_id,
        sync_module_states=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


class ModelFactory(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def get_model(self, *, local_rank: int, device_id: Union[int, Literal["cpu"]], world_size: int) -> Any:
        if device_id == "cpu":
            assert world_size == 1, "CPU offload only supports single-GPU inference"


class T5ModelFactory(ModelFactory):
    def __init__(self):
        super().__init__()

    def get_model(self, *, local_rank, device_id, world_size):
        super().get_model(local_rank=local_rank, device_id=device_id, world_size=world_size)
        model = T5EncoderModel.from_pretrained(T5_MODEL)
        if world_size > 1:
            model = setup_fsdp_sync(
                model,
                device_id=device_id,
                param_dtype=torch.float32,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={
                        T5Block,
                    },
                ),
            )
        elif isinstance(device_id, int):
            model = model.to(torch.device(f"cuda:{device_id}"))  # type: ignore
        return model.eval()


class DitModelFactory(ModelFactory):
    def __init__(self, *, model_path: str, model_dtype: str, attention_mode: Optional[str] = None):
        if attention_mode is None:
            from genmo.lib.attn_imports import flash_varlen_qkvpacked_attn  # type: ignore

            # attention_mode = "flex"
            attention_mode = "sdpa" if flash_varlen_qkvpacked_attn is None else "flash"
        print(f"Attention mode: {attention_mode}")
        super().__init__(
            model_path=model_path, model_dtype=model_dtype, attention_mode=attention_mode
        )

    def get_model(self, *, local_rank, device_id, world_size):
        # TODO(ved): Set flag for torch.compile
        from genmo.mochi_preview.dit.joint_model.asymm_models_joint import (
            AsymmDiTJoint,
        )

        model: nn.Module = torch.nn.utils.skip_init(
            AsymmDiTJoint,
            depth=48,
            patch_size=2,
            num_heads=24,
            hidden_size_x=3072,
            hidden_size_y=1536,
            mlp_ratio_x=4.0,
            mlp_ratio_y=4.0,
            in_channels=12,
            qk_norm=True,
            qkv_bias=False,
            out_bias=True,
            patch_embed_bias=True,
            timestep_mlp_bias=True,
            timestep_scale=1000.0,
            t5_feat_dim=4096,
            t5_token_length=256,
            rope_theta=10000.0,
            attention_mode=self.kwargs["attention_mode"],
        )

        if local_rank == 0:
            # FSDP syncs weights from rank 0 to all other ranks
            model.load_state_dict(load_file(self.kwargs["model_path"]))

        if world_size > 1:
            assert self.kwargs["model_dtype"] == "bf16", "FP8 is not supported for multi-GPU inference"
            model = setup_fsdp_sync(
                model,
                device_id=device_id,
                param_dtype=torch.bfloat16,
                auto_wrap_policy=partial(
                    lambda_auto_wrap_policy,
                    lambda_fn=lambda m: m in model.blocks,
                ),
            )
        elif isinstance(device_id, int):
            model = model.to(torch.device(f"cuda:{device_id}"))
        return model.eval()


class DecoderModelFactory(ModelFactory):
    def __init__(self, *, model_path: str):
        super().__init__(model_path=model_path)

    def get_model(self, *, local_rank, device_id, world_size):
        # TODO(ved): Set flag for torch.compile
        # TODO(ved): Use skip_init

        decoder = Decoder(
            out_channels=3,
            base_channels=128,
            channel_multipliers=[1, 2, 4, 6],
            temporal_expansions=[1, 2, 3],
            spatial_expansions=[2, 2, 2],
            num_res_blocks=[3, 3, 4, 6, 3],
            latent_dim=12,
            has_attention=[False, False, False, False, False],
            output_norm=False,
            nonlinearity="silu",
            output_nonlinearity="silu",
            causal=True,
        )
        # VAE is not FSDP-wrapped
        state_dict = load_file(self.kwargs["model_path"])
        decoder.load_state_dict(state_dict, strict=True)
        device = torch.device(f"cuda:{device_id}") if isinstance(device_id, int) else "cpu"
        decoder.eval().to(device)
        return decoder


def get_conditioning(tokenizer, encoder, device, batch_inputs, *, prompt: str, negative_prompt: str):
    if batch_inputs:
        return dict(batched=get_conditioning_for_prompts(tokenizer, encoder, device, [prompt, negative_prompt]))
    else:
        cond_input = get_conditioning_for_prompts(tokenizer, encoder, device, [prompt])
        null_input = get_conditioning_for_prompts(tokenizer, encoder, device, [negative_prompt])
        return dict(cond=cond_input, null=null_input)


def get_conditioning_for_prompts(tokenizer, encoder, device, prompts: List[str]):
    assert len(prompts) in [1, 2]  # [neg] or [pos] or [pos, neg]
    B = len(prompts)
    t5_toks = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=MAX_T5_TOKEN_LENGTH,
        return_tensors="pt",
        return_attention_mask=True,
    )
    caption_input_ids_t5 = t5_toks["input_ids"]
    caption_attention_mask_t5 = t5_toks["attention_mask"].bool()
    del t5_toks

    assert caption_input_ids_t5.shape == (B, MAX_T5_TOKEN_LENGTH)
    assert caption_attention_mask_t5.shape == (B, MAX_T5_TOKEN_LENGTH)

    # Special-case empty negative prompt by zero-ing it
    if prompts[-1] == "":
        caption_input_ids_t5[-1] = 0
        caption_attention_mask_t5[-1] = False

    caption_input_ids_t5 = caption_input_ids_t5.to(device, non_blocking=True)
    caption_attention_mask_t5 = caption_attention_mask_t5.to(device, non_blocking=True)

    y_mask = [caption_attention_mask_t5]
    y_feat = [encoder(caption_input_ids_t5, caption_attention_mask_t5).last_hidden_state.detach()]
    # Sometimes returns a tensor, othertimes a tuple, not sure why
    # See: https://huggingface.co/genmo/mochi-1-preview/discussions/3
    assert tuple(y_feat[-1].shape) == (B, MAX_T5_TOKEN_LENGTH, 4096)
    assert y_feat[-1].dtype == torch.float32

    return dict(y_mask=y_mask, y_feat=y_feat)


def compute_packed_indices(
    device: torch.device, text_mask: torch.Tensor, num_latents: int
) -> Dict[str, Union[torch.Tensor, int]]:
    """
    Based on https://github.com/Dao-AILab/flash-attention/blob/765741c1eeb86c96ee71a3291ad6968cfbf4e4a1/flash_attn/bert_padding.py#L60-L80

    Args:
        num_latents: Number of latent tokens
        text_mask: (B, L) List of boolean tensor indicating which text tokens are not padding.

    Returns:
        packed_indices: Dict with keys for Flash Attention:
            - valid_token_indices_kv: up to (B * (N + L),) tensor of valid token indices (non-padding)
                                   in the packed sequence.
            - cu_seqlens_kv: (B + 1,) tensor of cumulative sequence lengths in the packed sequence.
            - max_seqlen_in_batch_kv: int of the maximum sequence length in the batch.
    """
    # Create an expanded token mask saying which tokens are valid across both visual and text tokens.
    PATCH_SIZE = 2
    num_visual_tokens = num_latents // (PATCH_SIZE**2)
    assert num_visual_tokens > 0

    mask = F.pad(text_mask, (num_visual_tokens, 0), value=True)  # (B, N + L)
    seqlens_in_batch = mask.sum(dim=-1, dtype=torch.int32)  # (B,)
    valid_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()  # up to (B * (N + L),)
    assert valid_token_indices.size(0) >= text_mask.size(0) * num_visual_tokens  # At least (B * N,)
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen_in_batch = seqlens_in_batch.max().item()

    return {
        "cu_seqlens_kv": cu_seqlens.to(device, non_blocking=True),
        "max_seqlen_in_batch_kv": cast(int, max_seqlen_in_batch),
        "valid_token_indices_kv": valid_token_indices.to(device, non_blocking=True),
    }


def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def sample_model(device, dit, conditioning, **args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    generator = torch.Generator(device=device)
    generator.manual_seed(args["seed"])

    w, h, t = args["width"], args["height"], args["num_frames"]
    sample_steps = args["num_inference_steps"]
    cfg_schedule = args["cfg_schedule"]
    sigma_schedule = args["sigma_schedule"]

    assert_eq(len(cfg_schedule), sample_steps, "cfg_schedule must have length sample_steps")
    assert_eq((t - 1) % 6, 0, "t - 1 must be divisible by 6")
    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = 1
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 6
    IN_CHANNELS = 12
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    z = torch.randn(
        (B, IN_CHANNELS, latent_t, latent_h, latent_w),
        device=device,
        dtype=torch.float32,
    )

    num_latents = latent_t * latent_h * latent_w
    cond_batched = cond_text = cond_null = None
    if "cond" in conditioning:
        cond_text = conditioning["cond"]
        cond_null = conditioning["null"]
        cond_text["packed_indices"] = compute_packed_indices(device, cond_text["y_mask"][0], num_latents)
        cond_null["packed_indices"] = compute_packed_indices(device, cond_null["y_mask"][0], num_latents)
    else:
        cond_batched = conditioning["batched"]
        cond_batched["packed_indices"] = compute_packed_indices(device, cond_batched["y_mask"][0], num_latents)
        z = repeat(z, "b ... -> (repeat b) ...", repeat=2)

    def model_fn(*, z, sigma, cfg_scale, figure_mode=False):
        if cond_batched:
            assert False, "Should not reach here"
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = dit(z, sigma, **cond_batched)
            out_cond, out_uncond = torch.chunk(out, chunks=2, dim=0)
        else:
            nonlocal cond_text, cond_null

            mode = args["mode"]
            stg_block_idx = args["stg_block_idx"]
            stg_scale = args["stg_scale"]

            if mode == "STG":
                print(f"[INFO] ## STG Mode")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out_cond = dit(z, sigma, False, -1, **cond_text)
                    out_uncond = dit(z, sigma, False, -1, **cond_null)
                    out_perturb = dit(z, sigma, True, stg_block_idx, **cond_text)
            elif mode == "CFG":
                print(f"[INFO] ## CFG Mode")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out_cond = dit(z, sigma, False, -1, **cond_text)
                    out_uncond = dit(z, sigma, False, -1, **cond_null)
            elif mode == "Uncond":
                print(f"[INFO] ## Uncond Mode")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out_uncond = dit(z, sigma, False, -1, **cond_null)
            elif mode == "PASS":
                print(f"[INFO] ## PASS Mode")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out_cond = dit(z, sigma, False, -1, **cond_text)
                    out_uncond = dit(z, sigma, False, -1, **cond_null)
                    out_perturb = dit(z, sigma, True, stg_block_idx, **cond_text)
            elif mode == "perturb_STG":
                print(f"[INFO] ## perturb_STG Mode")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # out_cond = dit(z, sigma, False, -1, **cond_text)
                    # out_uncond = dit(z, sigma, False, -1, **cond_null)
                    out_perturb = dit(z, sigma, True, stg_block_idx, **cond_text)
            elif mode == "perturb_PASS":
                print(f"[INFO] ## perturb_PASS Mode")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # out_cond = dit(z, sigma, False, -1, **cond_text)
                    # out_uncond = dit(z, sigma, False, -1, **cond_null)
                    out_perturb = dit(z, sigma, True, stg_block_idx, **cond_text)
            else:
                assert False, "Wrong Mode"

        if mode == "Uncond":
            out_uncond = out_uncond.to(z)
        elif mode == "perturb_STG" or mode == "perturb_PASS":
            out_perturb = out_perturb.to(z)
            pass
        else:
            assert out_cond.shape == out_uncond.shape
            out_uncond = out_uncond.to(z)
            out_cond = out_cond.to(z)
        
        if mode == "STG" or mode == "PASS":
            print(f"[INFO] cfg_scale: {cfg_scale}")
            print(f"[INFO] stg_scale: {stg_scale}")
            out_perturb = out_perturb.to(z)

            if figure_mode:
                return out_uncond, out_cond, out_perturb
                
            #---------------------Rescaling----------------------#
            if args["do_rescaling"]:
                assert not figure_mode
                rescaling_scale = args["rescaling_scale"]
                print(f"[INFO] Rescaled with scale: {rescaling_scale}")
                output = out_uncond + cfg_scale * (out_cond - out_uncond) \
                              + stg_scale * (out_cond - out_perturb)
                factor = out_cond.std() / output.std()
                factor = rescaling_scale * factor + (1 - rescaling_scale)
                output = output * factor
                return output
            #----------------------------------------------------#

            return out_uncond + cfg_scale * (out_cond - out_uncond) \
                              + stg_scale * (out_cond - out_perturb)
        elif mode == "CFG":
            print(f"[INFO] cfg_scale: {cfg_scale}")
        #---------------------Rescaling----------------------#
            if args["do_rescaling"]:
                raise NotImplementedError
        #----------------------------------------------------#
            return out_uncond + cfg_scale * (out_cond - out_uncond)
        elif mode == "Uncond":
            return out_uncond
        elif mode == "perturb_STG":
            return out_perturb
        elif mode == "perturb_PASS":
            return out_perturb

    figure_mode = args["figure_mode"]
    figure_idx = args["figure_idx"]
    figure_start_idx = args["figure_start_idx"]
    latent_dir = "/scratch/slurm-user25-kaist/user/kinamkim/models/figure/latents"
    os.makedirs(latent_dir, exist_ok=True)
    figure_latent_path = f"/scratch/slurm-user25-kaist/user/kinamkim/models/figure/latents/latents_{figure_start_idx}.pt"
    latent_exist = os.path.exists(figure_latent_path)

    if figure_mode and latent_exist:
        z = torch.load(figure_latent_path)
        print(f"Loaded latents at step {figure_start_idx}")
    # Euler sampler w/ customizable sigma schedule & cfg scale
    for i in get_new_progress_bar(range(0, sample_steps), desc="Sampling"):
        if figure_mode and latent_exist and i < figure_start_idx:
            continue
        if figure_mode:
            latent_path = f"/scratch/slurm-user25-kaist/user/kinamkim/models/figure/latents/latents_{i}.pt"
            if not os.path.exists(latent_path):
                torch.save(z, latent_path)
                print(f"Latents at step {i} saved")

        local_rank = args["local_rank"]
        print(f"i: {i}") if local_rank == 0 else None
        sigma = sigma_schedule[i]
        dsigma = sigma - sigma_schedule[i + 1]

        if figure_mode and i == figure_idx:
            print(f"## Figure Mode")
            out_uncond, out_cond, out_perturb = model_fn(
                z=z,
                sigma=torch.full([B] if cond_text else [B * 2], sigma, device=z.device),
                cfg_scale=cfg_schedule[i],
                figure_mode=True,
            )
            stg_scale = args["stg_scale"]
            full_dsigma = sigma - sigma_schedule[-1]
            pred_z0_org = z
            pred_z0_cond = z + full_dsigma * out_cond
            pred_z0_uncond = z + full_dsigma * out_uncond
            pred_z0_perturb = z + full_dsigma * out_perturb
            pred_z0_cfg = z + full_dsigma * (out_uncond + cfg_schedule[i] * (out_cond - out_uncond))
            pred_z0_pass = z + full_dsigma * (out_uncond + stg_scale * (out_cond - out_perturb))
            pred_z0_total = z + full_dsigma * (out_uncond + cfg_schedule[i] * (out_cond - out_uncond) +
                                                + stg_scale * (out_cond - out_perturb))
            
            pred_z0_perturb_diff = z + full_dsigma * (out_cond - out_perturb)
            pred_z0_uncond_diff = z + full_dsigma * (out_cond - out_perturb)

            # pred_z0_list = [pred_z0_perturb, pred_z0_uncond, pred_z0_cond, pred_z0_pass, pred_z0_cfg, pred_z0_total]
            pred_z0_list = [pred_z0_pass]
            assert not cond_batched

            result_list = []
            # print(f"pred_zo_list length: {len(pred_z0_list)}")
            for z in pred_z0_list:
                z = z.detach()  # 그래프 연결 방지
                with torch.no_grad():  # 추적 방지
                    result = dit_latents_to_vae_latents(z)  # 처리
                    result_list.append(result)  # 처리 결과 저장

                # 처리 후 GPU 메모리 해제
                del z
                torch.cuda.empty_cache()  # GPU 캐시 비우기
            return result_list

        # `pred` estimates `z_0 - eps`.
        pred = model_fn(
            z=z,
            sigma=torch.full([B] if cond_text else [B * 2], sigma, device=z.device),
            cfg_scale=cfg_schedule[i],
        )
        assert pred.dtype == torch.float32
        z = z + dsigma * pred

        #------------------------RESTART---------------------------#
        do_restart = args["do_restart"]
        restart_idx = args["restart_idx"]
        num_restarts = args["num_restarts"]
        max_idx = args["max_idx"]
        local_rank = args["local_rank"]
        num_intervals = args["num_intervals"]

        if do_restart:
            if i + 1 == restart_idx:
                print(f"--- RESTARTING at step {i + 1} ---") if local_rank == 0 else None
                for restart_iter in range(num_restarts):
                    print(f"[INFO] Restart Iteration {restart_iter}") if local_rank == 0 else None
                    restart_t_steps = generate_restart_t_steps(max_idx, restart_idx+1, num_intervals, local_rank)
                    z = add_noise_interval(z, sigma_schedule, restart_t_steps[-1], restart_t_steps[0])

                    for t in restart_t_steps[:-1]:
                        print(f"[INFO] Restart timestep {t}") if local_rank == 0 else None
                        sigma = sigma_schedule[t]
                        dsigma = sigma - sigma_schedule[t + 1]

                        pred = model_fn(
                            z=z,
                            sigma=torch.full([B] if cond_text else [B * 2], sigma, device=z.device),
                            cfg_scale=cfg_schedule[t],
                        )
                        assert pred.dtype == torch.float32
                        z = z + dsigma * pred
                print(f"RESTART Done") if local_rank == 0 else None

        #---------------------------------------------------------#

    z = z[:B] if cond_batched else z
    return dit_latents_to_vae_latents(z)


@contextmanager
def move_to_device(model: nn.Module, target_device):
    og_device = next(model.parameters()).device
    if og_device == target_device:
        print(f"move_to_device is a no-op model is already on {target_device}")
    else:
        print(f"moving model from {og_device} -> {target_device}")

    model.to(target_device)
    yield
    if og_device != target_device:
        print(f"moving model from {target_device} -> {og_device}")
    model.to(og_device)


def t5_tokenizer():
    return T5Tokenizer.from_pretrained(T5_MODEL, legacy=False)


class MochiSingleGPUPipeline:
    def __init__(
        self,
        *,
        text_encoder_factory: ModelFactory,
        dit_factory: ModelFactory,
        decoder_factory: ModelFactory,
        cpu_offload: Optional[bool] = False,
        decode_type: str = "full",
        decode_args: Optional[Dict[str, Any]] = None,
    ):
        self.device = torch.device("cuda:0")
        self.tokenizer = t5_tokenizer()
        t = Timer()
        self.cpu_offload = cpu_offload
        self.decode_args = decode_args or {}
        self.decode_type = decode_type
        init_id = "cpu" if cpu_offload else 0
        with t("load_text_encoder"):
            self.text_encoder = text_encoder_factory.get_model(
                local_rank=0,
                device_id=init_id,
                world_size=1,
            )
        with t("load_dit"):
            self.dit = dit_factory.get_model(local_rank=0, device_id=init_id, world_size=1)
        with t("load_vae"):
            self.decoder = decoder_factory.get_model(local_rank=0, device_id=init_id, world_size=1)
        t.print_stats()

    def __call__(self, batch_cfg, prompt, negative_prompt, **kwargs):
        with torch.inference_mode():
            print_max_memory = lambda: print(
                f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB"
            )
            print_max_memory()
            with move_to_device(self.text_encoder, self.device):
                conditioning = get_conditioning(
                    self.tokenizer,
                    self.text_encoder,
                    self.device,
                    batch_cfg,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                )
            print_max_memory()
            with move_to_device(self.dit, self.device):
                latents = sample_model(self.device, self.dit, conditioning, **kwargs)
            print_max_memory()
            with move_to_device(self.decoder, self.device):
                frames = (
                    decode_latents_tiled_full(self.decoder, latents, **self.decode_args)
                    if self.decode_type == "tiled_full"
                    else decode_latents_tiled_spatial(self.decoder, latents, **self.decode_args)
                    if self.decode_type == "tiled_spatial"
                    else decode_latents(self.decoder, latents)
                )
            print_max_memory()
            return frames.cpu().numpy()


### ALL CODE BELOW HERE IS FOR MULTI-GPU MODE ###


# In multi-gpu mode, all models must belong to a device which has a predefined context parallel group
# So it doesn't make sense to work with models individually
class MultiGPUContext:
    def __init__(
        self,
        *,
        text_encoder_factory,
        dit_factory,
        decoder_factory,
        device_id,
        local_rank,
        world_size,
    ):
        t = Timer()
        self.device = torch.device(f"cuda:{device_id}")
        print(f"Initializing rank {local_rank+1}/{world_size}")
        assert world_size > 1, f"Multi-GPU mode requires world_size > 1, got {world_size}"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = os.getenv('MASTER_PORT', '29500')
        with t("init_process_group"):
            dist.init_process_group(
                "nccl",
                rank=local_rank,
                world_size=world_size,
                device_id=self.device,  # force non-lazy init
            )
        pg = dist.group.WORLD
        cp.set_cp_group(pg, list(range(world_size)), local_rank)
        distributed_kwargs = dict(local_rank=local_rank, device_id=device_id, world_size=world_size)
        self.world_size = world_size
        self.tokenizer = t5_tokenizer()
        with t("load_text_encoder"):
            self.text_encoder = text_encoder_factory.get_model(**distributed_kwargs)
        with t("load_dit"):
            self.dit = dit_factory.get_model(**distributed_kwargs)
        with t("load_vae"):
            self.decoder = decoder_factory.get_model(**distributed_kwargs)
        self.local_rank = local_rank
        t.print_stats()

    def run(self, *, fn, **kwargs):
        return fn(self, **kwargs)


class MochiMultiGPUPipeline:
    def __init__(
        self,
        *,
        text_encoder_factory: ModelFactory,
        dit_factory: ModelFactory,
        decoder_factory: ModelFactory,
        world_size: int,
    ):
        ray.init()
        RemoteClass = ray.remote(MultiGPUContext)
        self.ctxs = [
            RemoteClass.options(num_gpus=1).remote(
                text_encoder_factory=text_encoder_factory,
                dit_factory=dit_factory,
                decoder_factory=decoder_factory,
                world_size=world_size,
                device_id=0,
                local_rank=i,
            )
            for i in range(world_size)
        ]
        for ctx in self.ctxs:
            ray.get(ctx.__ray_ready__.remote())

    def __call__(self, **kwargs):
        def sample(ctx, *, batch_cfg, prompt, negative_prompt, **kwargs):
            with progress_bar(type="ray_tqdm", enabled=ctx.local_rank == 0), torch.inference_mode():
                conditioning = get_conditioning(
                    ctx.tokenizer,
                    ctx.text_encoder,
                    ctx.device,
                    batch_cfg,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                )
                #--------------------RESTART---------------------#
                kwargs["local_rank"] = ctx.local_rank
                #------------------------------------------------#
                figure_mode = kwargs["figure_mode"] 

                if figure_mode:
                    frames_list = []
                    latents_list = sample_model(ctx.device, ctx.dit, conditioning=conditioning, **kwargs)
                    # print(f"latents list length: {len(latents_list)}")
                    for latents in latents_list:
                        # latents를 GPU에서 처리
                        latents = latents.detach()  # 그래프 연결 방지
                        with torch.no_grad():  # 추적 방지
                            frames = decode_latents(ctx.decoder, latents)  # 디코딩

                        # 결과를 CPU로 이동해 numpy 배열로 변환 후 저장
                        frames_list.append(frames.cpu().numpy())

                        # 메모리 해제
                        del frames, latents
                        torch.cuda.empty_cache()  # GPU 캐시 비우기
                    return frames_list

                latents = sample_model(ctx.device, ctx.dit, conditioning=conditioning, **kwargs)
                if ctx.local_rank == 0:
                    torch.save(latents, "latents.pt")
                frames = decode_latents(ctx.decoder, latents)
            return frames.cpu().numpy()

        return ray.get([ctx.run.remote(fn=sample, **kwargs, show_progress=i == 0) for i, ctx in enumerate(self.ctxs)])[
            0
        ]
