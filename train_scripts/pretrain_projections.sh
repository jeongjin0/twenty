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
echo "Multi-GPU: 2 GPUs"
echo "============================================"
echo ""

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Multi-GPU training (2 GPUs)
accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --multi_gpu \
    train_scripts/pretrain_projections.py \
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
echo "Visualizations saved to: ${OUTPUT_DIR}/visualizations/"
echo ""
echo "각 에폭마다 다음 파일들이 생성됩니다:"
echo "  - merged_comparison.png: GT merged vs 예측 merged 비교"
echo "  - layers_comparison.png: GT layers vs 재구성 layers 비교"
echo "  - stats.txt: Merge/Decompose loss per layer"
echo ""
echo "Next: Use in main training"
echo "  python train_scripts/train_layer_inpainting.py \\"
echo "    --config ${CONFIG} \\"
echo "    --pretrained_pixart ... \\"
echo "    --pretrained_projections ${OUTPUT_DIR}/checkpoints/epoch_${EPOCHS}.pth"
echo "============================================"
