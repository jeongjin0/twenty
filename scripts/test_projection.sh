#!/bin/bash

# Projection Test - Trained Model
# Tests if projections maintain layer order after training

# Default values (can be overridden by command line args)
CHECKPOINT="${1:-output/inpainting_v4/checkpoints/epoch_10_step_50000.pth}"
DATA_ROOTS="../data/mulan_coco"
OUTPUT_DIR="output/projection_test_trained"
NUM_SAMPLES=5

echo "============================================"
echo "Projection Test - TRAINED MODEL"
echo "============================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Usage: bash scripts/test_projection.sh [checkpoint_path]"
echo "============================================"

python scripts/test_projection.py \
    --checkpoint ${CHECKPOINT} \
    --data_roots ${DATA_ROOTS} \
    --output_dir ${OUTPUT_DIR} \
    --num_samples ${NUM_SAMPLES}
