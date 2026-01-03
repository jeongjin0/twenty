#!/bin/bash

# Pretrain Projections with Merge/Decompose Strategy
# Input Projection: 6 layers → merged image latent (alpha blending)
# Output Projection: merged image latent → 6 layers (decomposition)

CONFIG="configs/layer_inpainting_config.py"
OUTPUT_DIR="output/projection_pretrain"
EPOCHS=10
LR=1e-4
BATCH_SIZE=4
MERGE_WEIGHT=1.0
DECOMPOSE_WEIGHT=1.0

echo "============================================"
echo "Projection Pretraining"
echo "============================================"
echo "Strategy: Merge (compositing) ↔ Decompose (separation)"
echo ""
echo "Input Projection:"
echo "  6 layers → merged image latent"
echo "  Output = real image latent (PixArt's space!)"
echo ""
echo "Output Projection:"
echo "  merged image latent → 6 layers"
echo "  Decompose image into layers"
echo ""
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}"
echo "LR: ${LR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Loss weights: merge=${MERGE_WEIGHT}, decompose=${DECOMPOSE_WEIGHT}"
echo "============================================"
echo ""

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python train_scripts/pretrain_projections.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --merge_weight ${MERGE_WEIGHT} \
    --decompose_weight ${DECOMPOSE_WEIGHT}

echo ""
echo "============================================"
echo "Pretraining complete!"
echo "============================================"
echo "Pretrained weights: ${OUTPUT_DIR}/checkpoints/epoch_${EPOCHS}.pth"
echo ""
echo "Next: Use in main training"
echo "  python train_scripts/train_layer_inpainting.py \\"
echo "    --config ${CONFIG} \\"
echo "    --pretrained_pixart ... \\"
echo "    --pretrained_projections ${OUTPUT_DIR}/checkpoints/epoch_${EPOCHS}.pth"
echo "============================================"
