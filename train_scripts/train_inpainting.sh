#!/bin/bash

# Layer-wise Inpainting Training Script
# Pretrained PixArt path is configured in configs/layer_inpainting_config.py

export PYTHONPATH="${PYTHONPATH}:/workspace/twenty"

torchrun --nproc_per_node=1 --master_port=29501 \
    train_scripts/train_layer_inpainting.py \
    --config configs/layer_inpainting_config.py

# To resume training:
# torchrun --nproc_per_node=1 --master_port=29501 \
#     train_scripts/train_layer_inpainting.py \
#     --config configs/layer_inpainting_config.py \
#     --resume output/layer_inpainting_v1/checkpoints/epoch_X_step_Y.pth
