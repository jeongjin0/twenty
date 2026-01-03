#!/bin/bash

# Layer-wise Inpainting Inference Script (Using Dataset)
# This script loads a sample from the dataset and generates a masked layer
# File format: {image_id}-layer_{n}.png

# ============================================
# Configuration
# ============================================
CHECKPOINT="output/layer_inpainting_v3/checkpoints/epoch_9_step_45000.pth"
VAE_PATH="/workspace/twenty/PixArt-alpha/sd-vae-ft-ema"
T5_PATH="/workspace/twenty/PixArt-alpha"

# Dataset configuration
DATA_DIR="/workspace/data/mulan_coco"  # or data/mulan_laion
IMAGE_ID="000000581346"  # Example image ID

# Which layer to mask and regenerate (0-5 for max_layers=6)
MASKED_IDX=2

# Text prompt (you can customize this or read from metadata)
PROMPT="a zebra standing on a black background"

# Inference parameters
# CRITICAL: Model was NOT trained with unconditional (null text)!
# CFG requires unconditional training to work properly.
# Using CFG without unconditional training causes colorful noise.
# Solution: Set CFG_SCALE=1.0 (no CFG, conditional only)
CFG_SCALE=1.0  # Was 4.5, but model doesn't support CFG yet
STEPS=50
MAX_LAYERS=6
IMAGE_SIZE=256

# Output
OUTPUT_DIR="output/layer_inpainting_v1/inference/${IMAGE_ID}"
mkdir -p ${OUTPUT_DIR}

# ============================================
# Extract visible layers from dataset
# ============================================
echo "============================================"
echo "Layer-wise Inpainting from Dataset"
echo "============================================"
echo "Image ID: ${IMAGE_ID}"
echo "Data directory: ${DATA_DIR}"
echo ""

# Find all layer files for this image ID
# Format: {image_id}-layer_{n}.png
# Use version sort (-V) for proper numeric ordering (layer_0, layer_1, ..., layer_10)
LAYER_FILES=($(ls ${DATA_DIR}/${IMAGE_ID}-layer_*.png 2>/dev/null | sort -V))
NUM_LAYERS=${#LAYER_FILES[@]}

if [ ${NUM_LAYERS} -eq 0 ]; then
    echo "Error: No layer files found for image ID ${IMAGE_ID}"
    echo "Looking for: ${DATA_DIR}/${IMAGE_ID}-layer_*.png"
    exit 1
fi

echo "Found ${NUM_LAYERS} layers:"
for f in "${LAYER_FILES[@]}"; do
    echo "  - $(basename $f)"
done
echo ""

# Build visible layers list (exclude masked layer)
VISIBLE_LAYERS=""
MASKED_LAYER=""

for i in "${!LAYER_FILES[@]}"; do
    if [ $i -eq ${MASKED_IDX} ]; then
        MASKED_LAYER="${LAYER_FILES[$i]}"
        echo "Layer $i: [MASKED] $(basename ${MASKED_LAYER})"
    else
        VISIBLE_LAYERS="${VISIBLE_LAYERS} ${LAYER_FILES[$i]}"
        echo "Layer $i: [VISIBLE] $(basename ${LAYER_FILES[$i]})"
    fi
done

if [ -z "${MASKED_LAYER}" ]; then
    echo "Error: Masked layer index ${MASKED_IDX} not found (only ${NUM_LAYERS} layers available)"
    exit 1
fi

echo ""
echo "Prompt: ${PROMPT}"
echo "============================================"
echo ""

# ============================================
# Run Inference
# ============================================
export PYTHONPATH="${PYTHONPATH}:/workspace/twenty"

OUTPUT_FILE="${OUTPUT_DIR}/generated_layer_${MASKED_IDX}.png"

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

# ============================================
# Create comparison (optional)
# ============================================
echo ""
echo "Creating comparison with ground truth..."

# Copy ground truth for comparison
if [ -f "${MASKED_LAYER}" ]; then
    GT_FILE="${OUTPUT_DIR}/ground_truth_layer_${MASKED_IDX}.png"
    cp "${MASKED_LAYER}" "${GT_FILE}"
    echo "  ✓ Ground truth: ${GT_FILE}"
fi

echo ""
echo "============================================"
echo "Inference complete!"
echo "============================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo "  - Generated: generated_layer_${MASKED_IDX}.png"
echo "  - All layers: generated_layer_${MASKED_IDX}_all_layers.png"
echo "  - Ground truth: ground_truth_layer_${MASKED_IDX}.png"
echo "============================================"
