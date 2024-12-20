import torch
import imageio

from pipeline_stg_stable_video_diffusion import StableVideoDiffusionSTGPipeline
# from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video


model_id = "/home/nas2_userG/junhahyung/kkn/checkpoint/stable-video-diffusion"

pipe = StableVideoDiffusionSTGPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, variant="fp16"
)
# pipe.enable_model_cpu_offload()
pipe = pipe.to("cuda")

# Load the conditioning image
image = load_image("https://raw.githubusercontent.com/junhahyung/diffusers/dev/workspace/stable_video_diffusion/assets/sample.png")
image = image.resize((1024, 512))

generator = torch.manual_seed(42)
frames = pipe(image, 
              decode_chunk_size=8, 
              generator=generator,
              stg_mode="STG-A",
              spatial_stg_applied_layers_idx=["u0"],
              temporal_stg_applied_layers_idx=["u0"],
              spatial_stg_scale=2.0, # 0.0 (default)
              temporal_stg_scale=0.5, # 0.0 (default)
              do_rescaling=True, # False (default)
              ).frames[0]

export_to_video(frames, "generated.mp4", fps=7)