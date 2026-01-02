#!/bin/bash

# Layer-wise Inpainting Inference Script
# This script generates a missing layer given visible layers

# ============================================
# Configuration
# ============================================
CHECKPOINT="output/layer_inpainting_v1/checkpoints/epoch_7_step_35000.pth"
VAE_PATH="/workspace/twenty/PixArt-alpha/sd-vae-ft-ema"
T5_PATH="/workspace/twenty/PixArt-alpha"

# Example: Using custom images
# Uncomment and set your own image paths
# VISIBLE_LAYER_1="/path/to/your/layer1.png"
# VISIBLE_LAYER_2="/path/to/your/layer2.png"

# Example: Using dataset images (mulan_coco format: {image_id}-layer_{n}.png)
DATA_DIR="/workspace/data/mulan_coco"
IMAGE_ID="000000581346"

# Build layer paths from dataset
LAYER_0="${DATA_DIR}/${IMAGE_ID}-layer_0.png"
LAYER_1="${DATA_DIR}/${IMAGE_ID}-layer_1.png"
LAYER_2="${DATA_DIR}/${IMAGE_ID}-layer_2.png"

# Use layers 0 and 2 as visible, regenerate layer 1
VISIBLE_LAYERS="${LAYER_0} ${LAYER_2}"
MASKED_IDX=3

# Text prompt for the layer to generate
PROMPT="a red apple on a white table"

# Output path
OUTPUT_DIR="output/layer_inpainting_v1/inference/${IMAGE_ID}"
OUTPUT_FILE="${OUTPUT_DIR}/generated_layer_${MASKED_IDX}.png"

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

echo "============================================"
echo "Layer-wise Inpainting Inference"
echo "============================================"
echo "Image ID: ${IMAGE_ID}"
echo "Visible layers: ${VISIBLE_LAYERS}"
echo "Generating layer: ${MASKED_IDX}"
echo "Prompt: ${PROMPT}"
echo "============================================"
echo ""

python scripts/infer_layer_inpainting.py \
    --checkpoint ${CHECKPOINT} \
    --visible_layers ${VISIBLE_LAYERS} \
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
echo "Inference complete! Results saved to:"
echo "  Generated: ${OUTPUT_FILE}"
echo "  All layers: ${OUTPUT_DIR}/generated_layer_${MASKED_IDX}_all_layers.png"
