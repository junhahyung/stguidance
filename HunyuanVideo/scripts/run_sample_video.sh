#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

python3 sample_video.py \
    --video-size 720 1280 \
    --video-length 129 \
    --prompt "A cat walks on the grass, realistic style." \
    --embedded-cfg-scale 6.0 \
    --infer-steps 50 \
    --flow-shift 7.0 \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./results \
