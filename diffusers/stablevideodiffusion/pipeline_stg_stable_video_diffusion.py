# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.models.attention_processor import Attention
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_video_diffusion import StableVideoDiffusionPipeline
import torch.nn.functional as F

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import StableVideoDiffusionPipeline
        >>> from diffusers.utils import load_image, export_to_video

        >>> pipe = StableVideoDiffusionPipeline.from_pretrained(
        ...     "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        ... )
        >>> pipe.to("cuda")

        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd-docstring-example.jpeg"
        ... )
        >>> image = image.resize((1024, 576))

        >>> frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        >>> export_to_video(frames, "generated.mp4", fps=7)
        ```
"""
class STGAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, mode="STG-A", temporal=False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.mode = mode
        self.temporal = temporal

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        assert encoder_hidden_states is None
        encoder_hidden_states_org, encoder_hidden_states_ptb = None, None
        hidden_states_uncond, hidden_states_org, hidden_states_spatial_ptb, hidden_states_temporal_ptb = hidden_states.chunk(4)
        if self.temporal:
            hidden_states_org = torch.cat([hidden_states_uncond, hidden_states_org, hidden_states_spatial_ptb])
            hidden_states_ptb = hidden_states_temporal_ptb
        else:
            hidden_states_org = torch.cat([hidden_states_uncond, hidden_states_org, hidden_states_temporal_ptb])
            hidden_states_ptb = hidden_states_spatial_ptb

        batch_size, sequence_length, _ = (
            hidden_states_org.shape if encoder_hidden_states_org is None else encoder_hidden_states_org.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states_org = attn.group_norm(hidden_states_org.transpose(1, 2)).transpose(1, 2)

        query_org = attn.to_q(hidden_states_org)

        if encoder_hidden_states_org is None:
            encoder_hidden_states_org = hidden_states_org
        elif attn.norm_cross:
            encoder_hidden_states_org = attn.norm_encoder_hidden_states_org(encoder_hidden_states_org)

        key_org = attn.to_k(encoder_hidden_states_org)
        value_org = attn.to_v(encoder_hidden_states_org)

        inner_dim = key_org.shape[-1]
        head_dim = inner_dim // attn.heads

        query_org = query_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key_org = key_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_org = value_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query_org = attn.norm_q(query_org)
        if attn.norm_k is not None:
            key_org = attn.norm_k(key_org)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states_org = F.scaled_dot_product_attention(
            query_org, key_org, value_org, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states_org = hidden_states_org.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query_org.dtype)

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states_org = hidden_states_org + residual

        hidden_states_org = hidden_states_org / attn.rescale_output_factor

        #--------------perturbed path--------------#
        if self.mode == "STG-A":
            batch_size, sequence_length, _ = (
                hidden_states_ptb.shape if encoder_hidden_states_ptb is None else encoder_hidden_states_ptb.shape
            )

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states_ptb = attn.group_norm(hidden_states_ptb.transpose(1, 2)).transpose(1, 2)

            query_ptb = attn.to_q(hidden_states_ptb)

            if encoder_hidden_states_ptb is None:
                encoder_hidden_states_ptb = hidden_states_ptb
            elif attn.norm_cross:
                encoder_hidden_states_ptb = attn.norm_encoder_hidden_states_ptb(encoder_hidden_states_ptb)

            key_ptb = attn.to_k(encoder_hidden_states_ptb)
            value_ptb = attn.to_v(encoder_hidden_states_ptb)

            inner_dim = key_ptb.shape[-1]
            head_dim = inner_dim // attn.heads

            query_ptb = query_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key_ptb = key_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_ptb = value_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query_ptb = attn.norm_q(query_ptb)
            if attn.norm_k is not None:
                key_ptb = attn.norm_k(key_ptb)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states_ptb = value_ptb

            hidden_states_ptb = hidden_states_ptb.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states_ptb = hidden_states_ptb.to(query_ptb.dtype)

            # linear proj
            hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
            # dropout
            hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

            if input_ndim == 4:
                hidden_states_ptb = hidden_states_ptb.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states_ptb = hidden_states_ptb + residual

            hidden_states_ptb = hidden_states_ptb / attn.rescale_output_factor
        #------------------------------------------#
        if self.temporal:
            hidden_states_uncond, hidden_states_org, hidden_states_spatial_ptb = hidden_states_org.chunk(3)
            hidden_states_temporal_ptb = hidden_states_ptb
        else:
            hidden_states_uncond, hidden_states_org, hidden_states_temporal_ptb = hidden_states_org.chunk(3)
            hidden_states_spatial_ptb = hidden_states_ptb
        hidden_states = torch.cat([hidden_states_uncond, hidden_states_org, hidden_states_spatial_ptb, hidden_states_temporal_ptb])

        return hidden_states

def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


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


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    r"""
    Output class for Stable Video Diffusion pipeline.

    Args:
        frames (`[List[List[PIL.Image.Image]]`, `np.ndarray`, `torch.Tensor`]):
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


class StableVideoDiffusionSTGPipeline(StableVideoDiffusionPipeline):
    def replace_layer_processor(self, down_layers, mid_layers, up_layers, applied_layers_index, replace_processor, initialize=False,):
        """
        Replace the processor for specified layers in the network.

        Args:
            layers (list): A list of tuples where each tuple contains (name, module) of the layer.
            applied_layers_index (list): A list of indices specifying which layers to replace.
            replace_processor (Callable): The processor to replace with.
        """
        if initialize:
            for dl in down_layers:
                dl[1].processor = replace_processor
            for ml in mid_layers:
                ml[1].processor = replace_processor
            for ul in up_layers:
                ul[1].processor = replace_processor
            print(f"[INFO] All layers are replaced with {replace_processor}")
        else:
            for drop_layer in applied_layers_index:
                try:
                    if drop_layer[0] == "d":
                        name, module = down_layers[int(drop_layer[1])]
                    elif drop_layer[0] == "m":
                        name, module = mid_layers[int(drop_layer[1])]
                    elif drop_layer[0] == "u":
                        name, module = up_layers[int(drop_layer[1])]
                    else:
                        raise ValueError(f"Invalid layer type: {drop_layer[0]}")
                    module.processor = replace_processor
                    print(f"[INFO] Layer {name} is replaced with {replace_processor}")

                except IndexError:
                    raise ValueError(
                        f"Invalid layer index: {drop_layer}. "
                    )
            
    def extract_layers(self, temporal=False, file_path="./unet_info.txt"):
        """
        Extract down, mid, and up layers from the UNet model and save to a file.

        Args:
            temporal (bool): Whether to extract temporal layers or not.
            file_path (str): The file path to save layer information.
        
        Returns:
            Tuple: (down_layers, mid_layers, up_layers)
        """
        down_layers, mid_layers, up_layers = [], [], []
        with open(file_path, "w") as f:
            for name, module in self.unet.named_modules():
                if "attn1" in name and "to" not in name:
                    if temporal:
                        if "temporal" in name:
                            f.write(f"{name}\n")
                            layer_type = name.split(".")[0].split("_")[0]
                            if layer_type == "down":
                                down_layers.append((name, module))
                            elif layer_type == "mid":
                                mid_layers.append((name, module))
                            elif layer_type == "up":
                                up_layers.append((name, module))
                            else:
                                raise ValueError(f"Invalid layer type: {layer_type}")
                    else:
                        if "temporal" not in name:
                            f.write(f"{name}\n")
                            layer_type = name.split(".")[0].split("_")[0]
                            if layer_type == "down":
                                down_layers.append((name, module))
                            elif layer_type == "mid":
                                mid_layers.append((name, module))
                            elif layer_type == "up":
                                up_layers.append((name, module))
                            else:
                                raise ValueError(f"Invalid layer type: {layer_type}")
        return down_layers, mid_layers, up_layers    
    @property
    def do_spatio_temporal_guidance(self):
        return self._spatial_stg_scale > 0.0 or self._temporal_stg_scale > 0.0

    def split_neg_org(self, x):
        split_point = x.size(0) // 2
        return x[:split_point], x[split_point:]

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        stg_mode: Optional[str] = "STG-R",
        spatial_stg_applied_layers_idx: Optional[List[int]] = ['u0'],
        spatial_stg_scale: Optional[float] = 1.0,
        temporal_stg_applied_layers_idx: Optional[List[int]] = ['u0'],
        temporal_stg_scale: Optional[float] = 1.0,
        do_rescaling: Optional[bool] = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.Tensor`):
                Image(s) to guide image generation. If you provide a tensor, the expected value range is between `[0,
                1]`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to `self.unet.config.num_frames` (14 for
                `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`).
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference. This parameter is modulated by `strength`.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after
                generation. Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                Used for conditioning the amount of motion for the generation. The higher the number the more motion
                will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the
                init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. Higher chunk size leads to better temporal consistency at the
                expense of more memory usage. By default, the decoder decodes all frames at once for maximal quality.
                For lower memory usage, reduce `decode_chunk_size`.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `pil`, `np` or `pt`.
            callback_on_step_end (`Callable`, *optional*):
                A function that is called at the end of each denoising step during inference. The function is called
                with the following arguments:
                    `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.
                `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is
                returned, otherwise a `tuple` of (`List[List[PIL.Image.Image]]` or `np.ndarray` or `torch.Tensor`) is
                returned.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale
        self._spatial_stg_scale = spatial_stg_scale
        self._temporal_stg_scale = temporal_stg_scale
        
        if self.do_spatio_temporal_guidance:
            down_layers, mid_layers, up_layers = self.extract_layers(temporal=False)
            replace_processor = STGAttnProcessor2_0(mode=stg_mode, temporal=False)
            self.replace_layer_processor(down_layers, 
                                        mid_layers, 
                                        up_layers, 
                                        spatial_stg_applied_layers_idx, 
                                        replace_processor,
                                        initialize=False)
            down_layers, mid_layers, up_layers = self.extract_layers(temporal=True)
            replace_processor = STGAttnProcessor2_0(mode=stg_mode, temporal=True)
            self.replace_layer_processor(down_layers,
                                        mid_layers,
                                        up_layers,
                                        temporal_stg_applied_layers_idx,
                                        replace_processor,
                                        initialize=False)

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)
        
        if self.do_spatio_temporal_guidance:
            neg_image_embeddings, org_image_embeddings = self.split_neg_org(image_embeddings)
            image_embeddings = torch.cat([neg_image_embeddings, org_image_embeddings, org_image_embeddings, org_image_embeddings])

        # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.video_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        
        if self.do_spatio_temporal_guidance:
            neg_image_latents, org_image_latents = self.split_neg_org(image_latents)
            image_latents = torch.cat([neg_image_latents, org_image_latents, org_image_latents, org_image_latents])
        
        image_latents = image_latents.to(image_embeddings.dtype)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        
        if self.do_spatio_temporal_guidance:
            added_time_ids, _ = self.split_neg_org(added_time_ids)
            added_time_ids = torch.cat([added_time_ids, added_time_ids, added_time_ids, added_time_ids])
        
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None, sigmas)

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 8. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                if self.do_classifier_free_guidance and not self.do_spatio_temporal_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                elif self.do_spatio_temporal_guidance:
                    latent_model_input = torch.cat([latents] * 4)
                else:
                    latent_model_input = latents
                
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Concatenate image_latents over channels dimension
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance and not self.do_spatio_temporal_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                elif self.do_spatio_temporal_guidance:
                    noise_pred_uncond, noise_pred_cond, noise_pred_ptb_spatial, noise_pred_ptb_temporal = noise_pred.chunk(4)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)\
                                    + self._spatial_stg_scale * (noise_pred_cond - noise_pred_ptb_spatial)\
                                    + self._temporal_stg_scale * (noise_pred_cond - noise_pred_ptb_temporal)

                if do_rescaling:
                    rescaling_scale = 0.7
                    factor = noise_pred_cond.std() / noise_pred.std()
                    factor = rescaling_scale * factor + (1 - rescaling_scale)
                    noise_pred = noise_pred * factor

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)


# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: List[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out