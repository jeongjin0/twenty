#!/bin/bash

# Layer-wise Inpainting Training Script

export PYTHONPATH="${PYTHONPATH}:/workspace/twenty"

# Pretrained PixArt checkpoint (from reference training or original)
PRETRAINED_PIXART="output/multilayer_pixart_256_clip_v3/checkpoints/epoch_7_step_60000.pth"

torchrun --nproc_per_node=1 --master_port=29501 \
    train_scripts/train_layer_inpainting.py \
    --config configs/layer_inpainting_config.py \
    --pretrained_pixart ${PRETRAINED_PIXART}

# To resume training:
# torchrun --nproc_per_node=1 --master_port=29501 \
#     train_scripts/train_layer_inpainting.py \
#     --config configs/layer_inpainting_config.py \
#     --pretrained_pixart ${PRETRAINED_PIXART} \
#     --resume output/layer_inpainting_v1/checkpoints/epoch_X_step_Y.pth
