# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import T5EncoderModel, T5TokenizerFast

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import Mochi1LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import MochiTransformer3DModel
from diffusers.models.attention_processor import Attention, MochiAttnProcessor2_0
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.mochi.pipeline_mochi import MochiPipeline
from diffusers.pipelines.mochi.pipeline_output import MochiPipelineOutput
import torch.nn.functional as F

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import MochiPipeline
        >>> from diffusers.utils import export_to_video

        >>> pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.bfloat16)
        >>> pipe.enable_model_cpu_offload()
        >>> pipe.enable_vae_tiling()
        >>> prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
        >>> frames = pipe(prompt, num_inference_steps=28, guidance_scale=3.5).frames[0]
        >>> export_to_video(frames, "mochi.mp4")
        ```
"""

class STGMochiAttnProcessor2_0:
    """Attention processor used in Mochi."""

    def __init__(self, mode="STG-R"):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("MochiAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        self.mode = mode

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states_neg, hidden_states_org, hidden_states_ptb = hidden_states.chunk(3)
        encoder_hidden_states_neg, encoder_hidden_states_org, encoder_hidden_states_ptb = encoder_hidden_states.chunk(3)
        
        hidden_states_org = torch.cat([hidden_states_neg, hidden_states_org], dim=0)
        encoder_hidden_states_org = torch.cat([encoder_hidden_states_neg, encoder_hidden_states_org], dim=0)
        
        query_org = attn.to_q(hidden_states_org)
        key_org = attn.to_k(hidden_states_org)
        value_org = attn.to_v(hidden_states_org)

        query_org = query_org.unflatten(2, (attn.heads, -1))
        key_org = key_org.unflatten(2, (attn.heads, -1))
        value_org = value_org.unflatten(2, (attn.heads, -1))

        if attn.norm_q is not None:
            query_org = attn.norm_q(query_org)
        if attn.norm_k is not None:
            key_org = attn.norm_k(key_org)

        encoder_query_org = attn.add_q_proj(encoder_hidden_states_org)
        encoder_key_org = attn.add_k_proj(encoder_hidden_states_org)
        encoder_value_org = attn.add_v_proj(encoder_hidden_states_org)

        encoder_query_org = encoder_query_org.unflatten(2, (attn.heads, -1))
        encoder_key_org = encoder_key_org.unflatten(2, (attn.heads, -1))
        encoder_value_org = encoder_value_org.unflatten(2, (attn.heads, -1))

        if attn.norm_added_q is not None:
            encoder_query_org = attn.norm_added_q(encoder_query_org)
        if attn.norm_added_k is not None:
            encoder_key_org = attn.norm_added_k(encoder_key_org)
        
        if image_rotary_emb is not None:

            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x_even = x[..., 0::2].float()
                x_odd = x[..., 1::2].float()

                cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
                sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)

                return torch.stack([cos, sin], dim=-1).flatten(-2)

            query_org = apply_rotary_emb(query_org, *image_rotary_emb)
            key_org = apply_rotary_emb(key_org, *image_rotary_emb)

        query_org, key_org, value_org = query_org.transpose(1, 2), key_org.transpose(1, 2), value_org.transpose(1, 2)
        encoder_query_org, encoder_key_org, encoder_value_org = (
            encoder_query_org.transpose(1, 2),
            encoder_key_org.transpose(1, 2),
            encoder_value_org.transpose(1, 2),
        )

        sequence_length_org = query_org.size(2)
        encoder_sequence_length_org = encoder_query_org.size(2)

        query_org = torch.cat([query_org, encoder_query_org], dim=2)
        key_org = torch.cat([key_org, encoder_key_org], dim=2)
        value_org = torch.cat([value_org, encoder_value_org], dim=2)

        hidden_states_org = F.scaled_dot_product_attention(query_org, key_org, value_org, dropout_p=0.0, is_causal=False)
        hidden_states_org = hidden_states_org.transpose(1, 2).flatten(2, 3)
        hidden_states_org = hidden_states_org.to(query_org.dtype)

        hidden_states_org, encoder_hidden_states_org = hidden_states_org.split_with_sizes(
            (sequence_length_org, encoder_sequence_length_org), dim=1
        )

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)

        if getattr(attn, "to_add_out", None) is not None:
            encoder_hidden_states_org = attn.to_add_out(encoder_hidden_states_org)

        #--------------perturb----------------#
        if self.mode == "STG-A":
            query_ptb = attn.to_q(hidden_states_ptb)
            key_ptb = attn.to_k(hidden_states_ptb)
            value_ptb = attn.to_v(hidden_states_ptb)

            query_ptb = query_ptb.unflatten(2, (attn.heads, -1))
            key_ptb = key_ptb.unflatten(2, (attn.heads, -1))
            value_ptb = value_ptb.unflatten(2, (attn.heads, -1))

            if attn.norm_q is not None:
                query_ptb = attn.norm_q(query_ptb)
            if attn.norm_k is not None:
                key_ptb = attn.norm_k(key_ptb)

            encoder_query_ptb = attn.add_q_proj(encoder_hidden_states_ptb)
            encoder_key_ptb = attn.add_k_proj(encoder_hidden_states_ptb)
            encoder_value_ptb = attn.add_v_proj(encoder_hidden_states_ptb)

            encoder_query_ptb = encoder_query_ptb.unflatten(2, (attn.heads, -1))
            encoder_key_ptb = encoder_key_ptb.unflatten(2, (attn.heads, -1))
            encoder_value_ptb = encoder_value_ptb.unflatten(2, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query_ptb = attn.norm_added_q(encoder_query_ptb)
            if attn.norm_added_k is not None:
                encoder_key_ptb = attn.norm_added_k(encoder_key_ptb)

            if image_rotary_emb is not None:
                query_ptb = apply_rotary_emb(query_ptb, *image_rotary_emb)
                key_ptb = apply_rotary_emb(key_ptb, *image_rotary_emb)

            query_ptb, key_ptb, value_ptb = query_ptb.transpose(1, 2), key_ptb.transpose(1, 2), value_ptb.transpose(1, 2)
            encoder_query_ptb, encoder_key_ptb, encoder_value_ptb = (
                encoder_query_ptb.transpose(1, 2),
                encoder_key_ptb.transpose(1, 2),
                encoder_value_ptb.transpose(1, 2),
            )

            sequence_length_ptb = query_ptb.size(2)
            encoder_sequence_length_ptb = encoder_query_ptb.size(2)

            query_ptb = torch.cat([query_ptb, encoder_query_ptb], dim=2)
            key_ptb = torch.cat([key_ptb, encoder_key_ptb], dim=2)
            value_ptb = torch.cat([value_ptb, encoder_value_ptb], dim=2)
            
            full_sequence_length_ptb = query_ptb.size(2)
            identity_block_size = query_ptb.size(2) - encoder_query_ptb.size(2)
            
            full_mask = torch.zeros((full_sequence_length_ptb, full_sequence_length_ptb), device=query_ptb.device, dtype=query_ptb.dtype)
            
            full_mask[:identity_block_size, :identity_block_size] = float("-inf")
            full_mask[:identity_block_size, identity_block_size:].fill_diagonal_(0)
            
            full_mask = full_mask.unsqueeze(0).unsqueeze(0)

            hidden_states_ptb = F.scaled_dot_product_attention(
                query_ptb, key_ptb, value_ptb, attn_mask=full_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states_ptb = hidden_states_ptb.transpose(1, 2).flatten(2, 3)
            hidden_states_ptb = hidden_states_ptb.to(query_ptb.dtype)

            hidden_states_ptb, encoder_hidden_states_ptb = hidden_states_ptb.split_with_sizes(
                (sequence_length_ptb, encoder_sequence_length_ptb), dim=1
            )

            # linear proj
            hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
            # dropout
            hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states_ptb = attn.to_add_out(encoder_hidden_states_ptb)

        #--------------------------------------#

        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb], dim=0)
        encoder_hidden_states = torch.cat([encoder_hidden_states_org, encoder_hidden_states_ptb], dim=0)

        return hidden_states, encoder_hidden_states

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# from: https://github.com/genmoai/models/blob/075b6e36db58f1242921deff83a1066887b9c9e1/src/mochi_preview/infer.py#L77
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
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class MochiSTGPipeline(MochiPipeline):
    def extract_layers(self, file_path="./unet_info.txt"):
        layers = []
        with open(file_path, "w") as f:
            for name, module in self.transformer.named_modules():
                if "attn1" in name and "to" not in name and "add" not in name and "norm" not in name:
                    f.write(f"{name}\n")
                    layer_type = name.split(".")[0].split("_")[0]
                    layers.append((name, module))

        return layers
    
    def replace_layer_processor(self, layers, replace_processor, stg_applied_layers_idx=[]):
        for layer_idx in stg_applied_layers_idx:
            layers[layer_idx][1].processor = replace_processor
            print(f"[INFO] Replaced {layer_idx}th layer with MochiSTGAttnProcessor2_0.")

        return
    
    @property
    def do_spatio_temporal_guidance(self):
        return self._stg_scale > 0.0

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 19,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        stg_mode: Optional[str] = "STG-R",
        stg_applied_layers_idx: Optional[List[int]] = [35],
        stg_scale: Optional[float] = 0.0,
        do_rescaling: Optional[bool] = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to `self.default_height`):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to `self.default_width`):
                The width in pixels of the generated image. This is set to 848 by default for the best results.
            num_frames (`int`, defaults to `19`):
                The number of video frames to generate
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, defaults to `4.5`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.mochi.MochiPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to `256`):
                Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.mochi.MochiPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.mochi.MochiPipelineOutput`] is returned, otherwise a `tuple`
                is returned where the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.default_height
        width = width or self.default_width

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        self._stg_scale = stg_scale
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if self.do_spatio_temporal_guidance:
            layers = self.extract_layers()
            replace_processor = STGMochiAttnProcessor2_0(mode=stg_mode)
            self.replace_layer_processor(layers, replace_processor, stg_applied_layers_idx)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Prepare text embeddings
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if self.do_classifier_free_guidance and not self.do_spatio_temporal_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        elif self.do_classifier_free_guidance and self.do_spatio_temporal_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask, prompt_attention_mask], dim=0)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timestep
        # from https://github.com/genmoai/models/blob/075b6e36db58f1242921deff83a1066887b9c9e1/src/mochi_preview/infer.py#L77
        threshold_noise = 0.025
        sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)
        sigmas = np.array(sigmas)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if self.do_classifier_free_guidance and not self.do_spatio_temporal_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                elif self.do_classifier_free_guidance and self.do_spatio_temporal_guidance:
                    latent_model_input = torch.cat([latents] * 3)
                    
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance and not self.do_spatio_temporal_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                elif self.do_classifier_free_guidance and self.do_spatio_temporal_guidance:
                    noise_pred_uncond, noise_pred_text, noise_pred_perturb = noise_pred.chunk(3)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond) \
                        + self._stg_scale * (noise_pred_text - noise_pred_perturb)
                        
                if do_rescaling:
                    rescaling_scale = 0.7
                    factor = noise_pred_text.std() / noise_pred.std()
                    factor = rescaling_scale * factor + (1 - rescaling_scale)
                    noise_pred = noise_pred * factor

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            video = latents
        else:
            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return MochiPipelineOutput(frames=video)
