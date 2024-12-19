import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from pipeline_stg_hunyuan import HunyuanVideoSTGPipeline
from diffusers.utils import export_to_video

model_id = "tencent/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16, revision='refs/pr/18'
)
pipe = HunyuanVideoSTGPipeline.from_pretrained(model_id, transformer=transformer, revision='refs/pr/18', torch_dtype=torch.float16)
pipe.vae.enable_tiling()
pipe.to("cuda")

#--------Option--------#
stg_mode = "STG-R"
stg_applied_layers_idx = [2]
stg_scale = 2.0
do_rescaling = True
#----------------------#

output = pipe(
    prompt="A massive dragon lands gently in a field of flowers, its scales shimmering in the sunlight, realistic style.",
    height=320,
    width=512,
    num_frames=61,
    num_inference_steps=30,
    stg_mode=stg_mode,
    stg_applied_layers_idx=stg_applied_layers_idx,
    stg_scale=stg_scale,
    do_rescaling=do_rescaling,
).frames[0]
export_to_video(output, "output.mp4", fps=15)
