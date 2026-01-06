#!/bin/bash

# Inpainting V4 - Simple Inference Script
# Generate a single layer from text prompt

# ============================================
# Configuration
# ============================================
CHECKPOINT="output/inpainting_v4/checkpoints/epoch_50.pth"  # Update this
PROMPT="a beautiful red rose"  # Your prompt here
OUTPUT="output/generated_layer.png"

# Optional: Add reference layers
# VISIBLE_LAYERS="layer0.png layer1.png"
VISIBLE_LAYERS=""
MASKED_IDX=0  # Which position to generate

# Generation settings
CFG_SCALE=1.0  # 1.0 = no CFG (recommended for models without unconditional training)
STEPS=50
IMAGE_SIZE=256
MAX_LAYERS=6

# Model paths
VAE_PATH="PixArt-alpha/sd-vae-ft-ema"
T5_PATH="PixArt-alpha"

# ============================================
# Run Inference
# ============================================
echo "============================================"
echo "Inpainting V4 Inference"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Prompt: '${PROMPT}'"
echo "  Output: ${OUTPUT}"
echo "  CFG Scale: ${CFG_SCALE}"
echo "  Steps: ${STEPS}"
echo ""
echo "Starting..."
echo "============================================"
echo ""

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Build command
CMD="python scripts/infer_v4.py \
    --checkpoint ${CHECKPOINT} \
    --prompt \"${PROMPT}\" \
    --output ${OUTPUT} \
    --cfg_scale ${CFG_SCALE} \
    --steps ${STEPS} \
    --image_size ${IMAGE_SIZE} \
    --max_layers ${MAX_LAYERS} \
    --vae_path ${VAE_PATH} \
    --t5_path ${T5_PATH} \
    --masked_idx ${MASKED_IDX}"

# Add visible layers if specified
if [ ! -z "$VISIBLE_LAYERS" ]; then
    CMD="${CMD} --visible_layers ${VISIBLE_LAYERS}"
fi

# Run
eval $CMD

echo ""
echo "============================================"
echo "Output saved to: ${OUTPUT}"
echo "============================================"
