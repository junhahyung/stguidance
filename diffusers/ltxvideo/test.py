import torch
from diffusers import LTXPipeline
from pipeline_stg_ltx import LTXSTGPipeline
from diffusers.utils import export_to_video

ckpt_path = "Lightricks/LTX-Video"
pipe = LTXSTGPipeline.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A woman with light skin, wearing a blue jacket and a black hat with a veil, looks down and to her right, then back up as she speaks; she has brown hair styled in an updo, light brown eyebrows, and is wearing a white collared shirt under her jacket; the camera remains stationary on her face as she speaks; the background is out of focus, but shows trees and people in period clothing; the scene is captured in real-life footage."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

stg_mode = "STG-R" # STG-A, STG-R, CFG
stg_applied_layers_idx = [19] # 0~27
stg_scale = 1.0 # 0.0 for CFG
do_rescaling = True # Default (False)

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=480,
    height=480,
    num_frames=81,
    num_inference_steps=50,
    generator=torch.manual_seed(42),
    stg_mode=stg_mode,
    stg_applied_layers_idx=stg_applied_layers_idx,
    stg_scale=stg_scale,
    do_rescaling=do_rescaling
).frames[0]
export_to_video(video, f"output.mp4", fps=24)