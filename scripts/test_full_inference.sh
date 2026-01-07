#!/bin/bash

# Full Inference Test
# Tests the actual denoising pipeline (not just projection reconstruction)

CHECKPOINT="${1:-output/inpainting_v4/checkpoints/epoch_10_step_50000.pth}"
DATA_ROOTS="../data/mulan_coco"
OUTPUT_DIR="output/full_inference_test"
NUM_SAMPLES=5
STEPS=50

echo "============================================"
echo "Full Inference Test"
echo "============================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Steps: ${STEPS}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "This tests ACTUAL inference:"
echo "  1. Add noise to one layer"
echo "  2. Run full model (input_proj → PixArt → output_proj)"
echo "  3. DDIM denoise"
echo "  4. Compare with ground truth"
echo ""
echo "Usage: bash scripts/test_full_inference.sh [checkpoint_path]"
echo "============================================"

python scripts/test_full_inference.py \
    --checkpoint ${CHECKPOINT} \
    --data_roots ${DATA_ROOTS} \
    --output_dir ${OUTPUT_DIR} \
    --num_samples ${NUM_SAMPLES} \
    --steps ${STEPS}
