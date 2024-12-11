import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_dir = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if args.prompt_path:
        save_dir = "./experiment"
    else:
        save_dir = "./results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    # Start sampling
    # TODO: batch inference check
    prompts_path = args.prompt_path
    prompts = []
    if prompts_path is not None:
        assert args.prompt is None, "Cannot specify both `prompt` and `prompts_path`"
        with open(prompts_path, 'r') as f:
            prompts = f.readlines()
        prompts = [p.strip() for p in prompts]
        
        for prompt in prompts:
            for idx in [0]:
                for embed_scale in [5.5, 5.6, 5.7, 5.8, 5.9]:
                    args.embedded_cfg_scale = embed_scale
                    args.stg_block_idx = idx
                    save_path = f"{save_dir}/{prompt[:100].replace('/','')}_seed{args.seed}"
                    os.makedirs(save_path, exist_ok=True)
                    if args.stg_mode:
                        save_path = f"{save_path}/{args.stg_mode}_block_{args.stg_block_idx}_scale_{args.stg_scale}_cfg_{args.cfg_scale}_embed_{args.embedded_cfg_scale}.mp4"
                    else:
                        save_path = f"{save_path}/NoSTG.mp4"
                    if os.path.exists(save_path):
                        print(f"Video Already Exists")
                        continue
                    outputs = hunyuan_video_sampler.predict(
                        prompt=prompt, 
                        height=args.video_size[0],
                        width=args.video_size[1],
                        video_length=args.video_length,
                        seed=args.seed,
                        negative_prompt=args.neg_prompt,
                        infer_steps=args.infer_steps,
                        guidance_scale=args.cfg_scale,
                        num_videos_per_prompt=args.num_videos,
                        flow_shift=args.flow_shift,
                        batch_size=args.batch_size,
                        embedded_guidance_scale=args.embedded_cfg_scale,
                        stg_mode=args.stg_mode,
                        stg_block_idx=args.stg_block_idx,
                        stg_scale=args.stg_scale,
                    )
                    samples = outputs['samples']
                    
                    # Save samples
                    for i, sample in enumerate(samples):
                        sample = samples[i].unsqueeze(0)
                        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
                        save_videos_grid(sample, save_path, fps=24)
                        logger.info(f'Sample save to: {save_path}')
    else:
        outputs = hunyuan_video_sampler.predict(
            prompt=args.prompt, 
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
            stg_mode=args.stg_mode,
            stg_block_idx=args.stg_block_idx,
            stg_scale=args.stg_scale,
        )
        samples = outputs['samples']
        
        # Save samples
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{save_dir}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, save_path, fps=24)
            logger.info(f'Sample save to: {save_path}')

if __name__ == "__main__":
    main()
