#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

nohup torchrun \
    --nproc_per_node=1 \
    --master_port=29501 \
    train_scripts/train_multilayer_ref_noalpha.py \
    configs/multilayer_config/PixArt_xl2_img256_multilayer_clip.py \
    --work-dir ./output/multilayer_pixart_256_clip_v3 \
    --use_ref True \
    --model_type crossattn \
    > train_clip_256_v3.log 2>&1 &

echo "PID: $!"
disown