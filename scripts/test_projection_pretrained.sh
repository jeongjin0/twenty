#!/bin/bash

# Projection Test - Pretrained Projections Only
# Tests the pretrained projection weights (before main training)

CHECKPOINT="output/projection_pretrain_v3/checkpoints/epoch_05.pth"
DATA_ROOTS="../data/mulan_coco"
OUTPUT_DIR="output/projection_test_pretrained"
NUM_SAMPLES=5

echo "============================================"
echo "Projection Test - PRETRAINED PROJECTIONS"
echo "============================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "This tests the projection weights BEFORE main training"
echo "to verify if layer order is preserved."
echo "============================================"

python scripts/test_projection_pretrained.py \
    --checkpoint ${CHECKPOINT} \
    --data_roots ${DATA_ROOTS} \
    --output_dir ${OUTPUT_DIR} \
    --num_samples ${NUM_SAMPLES}
