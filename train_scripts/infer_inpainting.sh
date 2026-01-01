#!/bin/bash

# Layer-wise Inpainting Inference Script
# This script generates a missing layer given visible layers

# ============================================
# Configuration
# ============================================
CHECKPOINT="output/layer_inpainting_v1/checkpoints/epoch_7_step_35000.pth"
VAE_PATH="/workspace/twenty/PixArt-alpha/sd-vae-ft-ema"
T5_PATH="/workspace/twenty/PixArt-alpha"

# Example input layers (replace with your own)
VISIBLE_LAYER_1="path/to/layer1.png"
VISIBLE_LAYER_2="path/to/layer2.png"
# Add more visible layers as needed

# Which layer index to generate (0-5 for max_layers=6)
MASKED_IDX=1

# Text prompt for the layer to generate
PROMPT="a red apple on a white table"

# Output path
OUTPUT_DIR="output/layer_inpainting_v1/inference"
OUTPUT_FILE="${OUTPUT_DIR}/generated_layer.png"

# Inference parameters
CFG_SCALE=4.5
STEPS=50
MAX_LAYERS=6
IMAGE_SIZE=256

# ============================================
# Run Inference
# ============================================
mkdir -p ${OUTPUT_DIR}

export PYTHONPATH="${PYTHONPATH}:/workspace/twenty"

python scripts/infer_layer_inpainting.py \
    --checkpoint ${CHECKPOINT} \
    --visible_layers ${VISIBLE_LAYER_1} ${VISIBLE_LAYER_2} \
    --masked_idx ${MASKED_IDX} \
    --prompt "${PROMPT}" \
    --output ${OUTPUT_FILE} \
    --cfg_scale ${CFG_SCALE} \
    --steps ${STEPS} \
    --max_layers ${MAX_LAYERS} \
    --image_size ${IMAGE_SIZE} \
    --vae_path ${VAE_PATH} \
    --t5_path ${T5_PATH}

echo ""
echo "Inference complete! Results saved to: ${OUTPUT_FILE}"
