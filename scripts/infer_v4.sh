#!/bin/bash

# Inpainting V4 - Dataset Inference
# Test layer inpainting with reference layers from MuLan dataset

# ============================================
# Configuration
# ============================================
CHECKPOINT="output/inpainting_v4/checkpoints/epoch_10_step_50000.pth"
DATA_ROOTS="../data/mulan_coco ../data/mulan_laion"
OUTPUT_DIR="output/inference_v4"

NUM_SAMPLES=10
CFG_SCALE=1.0  # 1.0 = no CFG (recommended)
STEPS=50
MAX_LAYERS=6
IMAGE_SIZE=256

VAE_PATH="PixArt-alpha/sd-vae-ft-ema"
T5_PATH="PixArt-alpha"

# ============================================
# Run Inference
# ============================================
echo "============================================"
echo "Inpainting V4 - Dataset Inference"
echo "============================================"
echo ""
echo "Configuration:"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Data: ${DATA_ROOTS}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Samples: ${NUM_SAMPLES}"
echo "  CFG Scale: ${CFG_SCALE}"
echo "  Steps: ${STEPS}"
echo ""
echo "This will:"
echo "  1. Load samples from MuLan dataset"
echo "  2. Use reference layers as conditioning"
echo "  3. Generate each layer and compare with GT"
echo ""
echo "Starting..."
echo "============================================"
echo ""

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python scripts/infer_v4.py \
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
echo "Inference complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Check the results:"
echo "  - generated.png: Generated layer"
echo "  - ground_truth.png: GT layer"
echo "  - comparison.png: Side-by-side comparison"
echo "  - all_layers.png: All layers (with generated)"
echo "  - all_layers_gt.png: All layers (ground truth)"
echo "============================================"
