#! /usr/bin/env python
import json
import os
import time

import click
import numpy as np
import torch

from genmo.lib.progress import progress_bar
from genmo.lib.utils import save_video
from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    MochiMultiGPUPipeline,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)

import random

pipeline = None
model_dir_path = None
num_gpus = torch.cuda.device_count()
cpu_offload = False


def configure_model(model_dir_path_, cpu_offload_):
    global model_dir_path, cpu_offload
    model_dir_path = model_dir_path_
    cpu_offload = cpu_offload_


def load_model():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = "29502"
    global num_gpus, pipeline, model_dir_path
    if pipeline is None:
        MOCHI_DIR = model_dir_path
        print(f"Launching with {num_gpus} GPUs. If you want to force single GPU mode use CUDA_VISIBLE_DEVICES=0.")
        klass = MochiSingleGPUPipeline if num_gpus == 1 else MochiMultiGPUPipeline
        kwargs = dict(
            text_encoder_factory=T5ModelFactory(),
            dit_factory=DitModelFactory(
                model_path=f"{MOCHI_DIR}/dit.safetensors", 
                model_dtype="bf16"
            ),
            decoder_factory=DecoderModelFactory(
                model_path=f"{MOCHI_DIR}/decoder.safetensors",
            ),
        )
        if num_gpus > 1:
            assert not cpu_offload, "CPU offload not supported in multi-GPU mode"
            kwargs["world_size"] = num_gpus
        else:
            kwargs["cpu_offload"] = cpu_offload
            kwargs["decode_type"] = "tiled_full"
        pipeline = klass(**kwargs)


def generate_video(
    prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    seed,
    cfg_scale,
    num_inference_steps,
    video_path,
    stg_block_idx,
    stg_scale,
    mode,
#--------------RESTART---------------#
    do_restart,
    restart_idx,
    num_restarts,
    max_idx,
    num_intervals,
#-------------Rescaling--------------#
    do_rescaling,
    rescaling_scale,
#------------------------------------#
):
    load_model()

    # sigma_schedule should be a list of floats of length (num_inference_steps + 1),
    # such that sigma_schedule[0] == 1.0 and sigma_schedule[-1] == 0.0 and monotonically decreasing.
    sigma_schedule = linear_quadratic_schedule(num_inference_steps, 0.025)

    # cfg_schedule should be a list of floats of length num_inference_steps.
    # For simplicity, we just use the same cfg scale at all timesteps,
    # but more optimal schedules may use varying cfg, e.g:
    # [5.0] * (num_inference_steps // 2) + [4.5] * (num_inference_steps // 2)
    cfg_schedule = [cfg_scale] * num_inference_steps

    args = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "sigma_schedule": sigma_schedule,
        "cfg_schedule": cfg_schedule,
        "num_inference_steps": num_inference_steps,
        # We *need* flash attention to batch cfg
        # and it's only worth doing in a high-memory regime (assume multiple GPUs)
        "batch_cfg": False,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "stg_block_idx": stg_block_idx,
        "stg_scale": stg_scale,
        "mode": mode,
        #---------RESTART-----------#
        "do_restart": do_restart,
        "restart_idx": restart_idx,
        "num_restarts": num_restarts,
        "max_idx": max_idx,
        "num_intervals": num_intervals,
        #---------------------------#
        #---------Rescaling---------#
        "do_rescaling": do_rescaling,
        "rescaling_scale": rescaling_scale,
        #---------------------------#
    }

    with progress_bar(type="tqdm"):
        final_frames = pipeline(**args)

        assert isinstance(final_frames, np.ndarray)
        assert final_frames.dtype == np.float32
        final_frames = final_frames[0]
        output_path=video_path
        save_video(final_frames, output_path)
        json_path = os.path.splitext(output_path)[0] + ".json"
        # json.dump(args, open(json_path, "w"), indent=4)

        return output_path

from textwrap import dedent

DEFAULT_PROMPT = dedent("""
A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl 
filled with lemons and sprigs of mint against a peach-colored background. 
The hand gently tosses the lemon up and catches it, showcasing its smooth texture. 
A beige string bag sits beside the bowl, adding a rustic touch to the scene. 
Additional lemons, one halved, are scattered around the base of the bowl. 
The even lighting enhances the vibrant colors and creates a fresh, 
inviting atmosphere.
""")

def generate_cli():
    import os
    import re
    import importlib

    import importlib.util
    import sys

    # Define the path to the config file
    config_path = "demos/config.py"

    # Get the absolute path to the config file
    config_file = os.path.abspath(config_path)

    # Check if the config file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Load the config file as a module
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Ensure only one of prompt or prompts_path is provided
    if not config_module.prompt and not config_module.prompts_path:
        raise ValueError("You must provide either 'prompt' or 'prompts_path' in the config.")

    if config_module.prompt and config_module.prompts_path:
        raise ValueError("You can provide only one of 'prompt' or 'prompts_path', not both.")

    # Get prompts from config
    if config_module.prompts_path:
        # Read prompts from the prompts.txt file
        with open(config_module.prompts_path, 'r') as f:
            prompts = f.readlines()

        # Remove any trailing newline characters from each prompt
        prompts = [prompt.strip() for prompt in prompts]
    else:
        # If prompt is provided, create a list with the prompt
        prompts = [config_module.prompt.strip()]

    # Function to sanitize filenames
    def sanitize_filename(filename):
        # Remove special characters and replace spaces with underscores
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = filename.replace(' ', '_')
        return filename[:50]

    negative_prompt = ""

    # Load configuration options from the config file
    mode = config_module.mode
    stg_block_idx = config_module.stg_block_idx
    stg_scale = config_module.stg_scale

    # Rescaling options
    do_rescaling = config_module.do_rescaling
    rescaling_scale = config_module.rescaling_scale

    # Restart options
    do_restart = config_module.do_restart
    restart_idx = config_module.restart_idx
    num_restarts = config_module.num_restarts
    max_idx = config_module.max_idx
    num_intervals = config_module.num_intervals

    # CFG Scale and Steps
    cfg_scale = config_module.cfg_scale
    num_steps = config_module.num_steps

    # Video generation options
    width = config_module.width
    height = config_module.height
    num_frames = config_module.num_frames
    seed = config_module.seed

    # Model directory and CPU offload
    model_dir = config_module.model_dir
    cpu_offload = config_module.cpu_offload

    # Configure the model using the provided config values
    configure_model(model_dir, cpu_offload)

    # Set the experiment directory and create directories for outputs
    experiment_dir = "outputs"
    mode_dir = os.path.join(experiment_dir, mode)

    # os.makedirs(mode_dir, exist_ok=True)

    # Iterate over prompts and generate videos
    for i, prompt in enumerate(prompts):
        sanitized_prompt = sanitize_filename(prompt)  # 파일 이름에 사용할 수 있도록 프롬프트를 변환
        for scale in stg_scale:
            for idx in stg_block_idx:
                if "STG-R" in mode:  # STG-R or perturb_PASS 모드 확인
                    curr_idx = f"residual_{idx}"
                else:
                    curr_idx = idx

                if mode == "STG-A" or mode == "STG-R":
                    stg_dir = os.path.join(mode_dir, f"idx_{idx}_scale_{scale}")
                    #----------RESTART----------#
                    if do_restart:
                        video_dir = f"{stg_dir}_restart_idx_{max_idx}-{restart_idx}_N_{num_restarts}_K_{num_intervals}"
                    else:
                        video_dir = stg_dir

                    if not do_rescaling:
                        os.makedirs(video_dir, exist_ok=True)
                    video_path = os.path.join(video_dir, f"{sanitized_prompt}.mp4")
                    #--------------------------#
                elif mode == "CFG":
                    #----------RESTART----------#
                    if do_restart:
                        restart_dir = f"{mode_dir}_restart_idx_{max_idx}-{restart_idx}_N_{num_restarts}_K_{num_intervals}"
                        video_dir = restart_dir
                    else:
                        video_dir = mode_dir
                    video_path = os.path.join(video_dir, f"{sanitized_prompt}.mp4")
                    if not do_rescaling:
                        os.makedirs(video_dir, exist_ok=True)
                    #--------------------------#

                #-------------Rescaling--------------#
                def modify_video_path(path, rescaling_scale):
                    video_dir, video_name = os.path.split(path)

                    parent_dir = os.path.dirname(video_dir)
                    last_dir = os.path.basename(video_dir)
                    new_dir = f"{last_dir}_rescaling_{rescaling_scale}"
                    new_video_dir = os.path.join(parent_dir, new_dir)
                    os.makedirs(new_video_dir, exist_ok=True)

                    new_video_path = os.path.join(new_video_dir, video_name)

                    return new_video_path

                if do_rescaling:
                    if not "rescale" in video_path:
                        video_path = modify_video_path(video_path, rescaling_scale)
                #------------------------------------#

                if os.path.exists(video_path):
                    print(f"[INFO] Skipping {video_path}...")
                    continue

                output = generate_video(
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    num_frames,
                    seed,
                    cfg_scale,
                    num_steps,
                    video_path,
                    curr_idx,
                    scale,
                    mode,
                    #----------RESTART----------#
                    do_restart,
                    restart_idx,
                    num_restarts,
                    max_idx,
                    num_intervals,
                    #---------Rescaling---------#
                    do_rescaling,
                    rescaling_scale,
                    #---------------------------#
                )
                click.echo(f"Video generated at: {video_path}")


if __name__ == "__main__":
    generate_cli()
