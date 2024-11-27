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
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '29500')
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
#------------------------------------#
    figure_mode,
    figure_idx,
    figure_start_idx,
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
        "figure_mode": figure_mode,
        "figure_idx": figure_idx,
        "figure_start_idx": figure_start_idx,
        #---------Rescaling---------#
        "do_rescaling": do_rescaling,
        "rescaling_scale": rescaling_scale,
        #---------------------------#
    }

    with progress_bar(type="tqdm"):
        final_frames = pipeline(**args)

        if figure_mode:
            print(f"Length: {len(final_frames)}")
            mode_list = ["perturb", "uncond", "cond", "pass", "cfg", "total", "diff_perturb", "diff_uncond"]
            mode_list = ["pass"]
            for i, frames in enumerate(final_frames):
                frames = frames[0]
                output_path = f"{video_path[:-4]}_idx_{figure_idx}_{mode_list[i]}.mp4"
                save_video(frames, output_path)
            frames_perturb_diff = final_frames[2][0] - final_frames[0][0]
            frames_uncond_diff = final_frames[2][0] - final_frames[1][0]
            output_path = f"{video_path[:-4]}_idx_{figure_idx}_diff_perturb.mp4"
            save_video(frames_perturb_diff, output_path)
            output_path = f"{video_path[:-4]}_idx_{figure_idx}_diff_uncond.mp4"
            save_video(frames_uncond_diff, output_path)
            assert False, "Stop here"
            return output_path

        assert isinstance(final_frames, np.ndarray)
        assert final_frames.dtype == np.float32
        final_frames = final_frames[0]
        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", f"output_{int(time.time())}.mp4")
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

@click.command()
@click.option("--prompt", default=DEFAULT_PROMPT, help="Prompt for video generation.")
@click.option("--negative_prompt", default="", help="Negative prompt for video generation.")
@click.option("--width", default=800, type=int, help="Width of the video.")
@click.option("--height", default=480, type=int, help="Height of the video.")
@click.option("--num_frames", default=91, type=int, help="Number of frames.")
@click.option("--seed", default=1710977262, type=int, help="Random seed.")
@click.option("--cfg_scale", default=4.5, type=float, help="CFG Scale.")
@click.option("--num_steps", default=64, type=int, help="Number of inference steps.")
@click.option("--model_dir", required=True, help="Path to the model directory.")
@click.option("--cpu_offload", is_flag=True, help="Whether to offload model to CPU")
def generate_cli(
    prompt, negative_prompt, width, height, num_frames, seed, cfg_scale, num_steps, model_dir, cpu_offload
):
    configure_model(model_dir, cpu_offload)

    import os
    import re

    def sanitize_filename(filename):
        # 특수 문자를 제거하고 공백을 밑줄로 대체
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = filename.replace(' ', '_')
        # 길이 제한 (필요에 따라 조정 가능)
        return filename[:50]

    # Block length: 48
    #--------------Option-------------#
    mode = "PASS" # CFG / STG / Uncond / PASS / perturb_STG / perturb_PASS
    # stg_block_idx=[30, 32, 35,]
    # stg_scale=[1.0, 2.0]
    stg_block_idx=[35]
    stg_scale=[0.8]
    #---------------------------------#
    #----------Rescaling--------#
    do_rescaling = True
    rescaling_scale = 0.7
    #---------------------------#
    #----------SAMPLE-----------#
    sample_mode = False
    #----------FIGURE-----------#
    figure_mode = False
    figure_idx = 37
    figure_start_idx = 37
    #----------RESTART----------#
    do_restart = False # True or False
    restart_idx = 63
    num_restarts = 1
    max_idx = 1
    num_intervals = -1
    #---------------------------#

    if mode == "CFG" or mode == "Uncond":
        stg_block_idx=[-1]
        stg_scale = [0.0]

    # Set paths and create directories
    # prompt_path = "/scratch/slurm-user25-kaist/user/kinamkim/models/filtered_quotes.txt"
    prompt_path = "/scratch/slurm-user25-kaist/user/kinamkim/models/prompt.txt"
    experiment_dir = "experiment"
    if figure_mode:
        prompt_path = "/scratch/slurm-user25-kaist/user/kinamkim/models/figure_prompt.txt"
        experiment_dir = "figure"
        
    if sample_mode:
        prompt_path = "/scratch/slurm-user25-kaist/user/kinamkim/models/sample_prompt.txt"
        experiment_dir = "sample"
        assert not figure_mode
        
    mode_dir = os.path.join(experiment_dir, mode)

    # Create experiment/CFG directories if they do not exist
    os.makedirs(mode_dir, exist_ok=True)

    # Read prompts from the prompts.txt file
    with open(prompt_path, 'r') as f:
        prompts = f.readlines()

    # Remove any trailing newline characters from each prompt
    prompts = [prompt.strip() for prompt in prompts]

    # Iterate over prompts and generate videos
    for i, prompt in enumerate(prompts):
        sanitized_prompt = sanitize_filename(prompt)  # 파일 이름에 사용할 수 있도록 프롬프트를 변환
        for scale in stg_scale:
            for idx in stg_block_idx:
                if "PASS" in mode:  # PASS or perturb_PASS 모드 확인
                    curr_idx = f"pass_{idx}"
                else:
                    curr_idx = idx

                if mode == "STG" or mode == "PASS":
                    stg_dir = os.path.join(mode_dir, f"idx_{idx}_scale_{scale}")
                    #----------RESTART----------#
                    if do_restart:
                        video_dir = f"{stg_dir}_restart_idx_{max_idx}-{restart_idx}_N_{num_restarts}_K_{num_intervals}"
                    else:
                        video_dir = stg_dir
                    # os.makedirs(stg_dir, exist_ok=True)
                    # video_path = os.path.join(stg_dir, f"{sanitized_prompt}.mp4")
                    os.makedirs(video_dir, exist_ok=True)
                    video_path = os.path.join(video_dir, f"{sanitized_prompt}.mp4")
                    #--------------------------#
                elif mode == "perturb_STG" or mode == "perturb_PASS":
                    stg_dir = os.path.join(mode_dir, f"idx_{idx}")
                    os.makedirs(stg_dir, exist_ok=True)
                    video_path = os.path.join(stg_dir, f"{sanitized_prompt}.mp4")
                elif mode == "CFG" or mode == "Uncond":
                    #----------RESTART----------#
                    if do_restart:
                        restart_dir = f"{mode_dir}_restart_idx_{max_idx}-{restart_idx}_N_{num_restarts}_K_{num_intervals}"
                        os.makedirs(restart_dir, exist_ok=True)
                        video_dir = restart_dir
                    else:
                        video_dir = mode_dir
                    # video_path = os.path.join(mode_dir, f"{sanitized_prompt}.mp4")
                    video_path = os.path.join(video_dir, f"{sanitized_prompt}.mp4")
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

                if os.path.exists(video_path) and not figure_mode:
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
                    #---------------------------#
                    figure_mode,
                    figure_idx,
                    figure_start_idx,
                    #---------Rescaling---------#
                    do_rescaling,
                    rescaling_scale,
                    #---------------------------#
                )
                click.echo(f"Video generated at: {video_path}")


if __name__ == "__main__":
    generate_cli()
