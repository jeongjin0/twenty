#!/bin/bash

# Inpainting V4 - Repaint-style Inference
# Fix mixed timestep issue by renoising visible layers

CHECKPOINT="output/inpainting_v4/checkpoints/epoch_10_step_50000.pth"
DATA_ROOTS="../data/mulan_coco ../data/mulan_laion"
OUTPUT_DIR="output/inference_v4_repaint"

NUM_SAMPLES=10
CFG_SCALE=1.0
STEPS=50
MAX_LAYERS=6
IMAGE_SIZE=256

VAE_PATH="PixArt-alpha/sd-vae-ft-ema"
T5_PATH="PixArt-alpha"

echo "============================================"
echo "Inpainting V4 - Repaint-style Inference"
echo "============================================"
echo ""
echo "FIX: Visible layer renoising"
echo "  Problem: Training uses same timestep for all layers"
echo "           Inference had mixed timesteps (visible=clean, masked=noisy)"
echo "  Solution: Renoise visible layers at each step to match masked timestep"
echo ""
echo "Configuration:"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Samples: ${NUM_SAMPLES}"
echo "  Steps: ${STEPS}"
echo "============================================"
echo ""

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python scripts/infer_v4_repaint.py \
    --checkpoint ${CHECKPOINT} \
    --data_roots ${DATA_ROOTS} \
    --output_dir ${OUTPUT_DIR} \
    --num_samples ${NUM_SAMPLES} \
    --cfg_scale ${CFG_SCALE} \
    --steps ${STEPS} \
    --max_layers ${MAX_LAYERS} \
    --image_size ${IMAGE_SIZE} \
    --vae_path ${VAE_PATH} \
    --t5_path ${T5_PATH}

echo ""
echo "============================================"
echo "Done! Compare with original inference:"
echo "  Original: output/inference_v4/"
echo "  Repaint:  ${OUTPUT_DIR}/"
echo "============================================"
