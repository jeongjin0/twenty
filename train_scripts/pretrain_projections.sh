#!/bin/bash

# Pretrain Projections with Merge/Decompose Strategy
# Input Projection: 6 layers → merged image latent (alpha blending)
# Output Projection: merged image latent → 6 layers (decomposition)

CONFIG="configs/layer_inpainting_config.py"
OUTPUT_DIR="output/projection_pretrain_v3"
EPOCHS=10
LR=1e-4
BATCH_SIZE=8
MERGE_WEIGHT=1.0
DECOMPOSE_WEIGHT=1.0
BACKGROUND_WEIGHT=0.1  # Layer 0 (background) much lower weight
TARGET_LAYER_WEIGHT=2.0  # Target foreground layer higher weight
ENABLE_SHUFFLE="--enable_shuffle"  # Shuffle input for order-invariance

echo "============================================"
echo "Projection Pretraining - Optimal Assignment"
echo "============================================"
echo "Strategy: Structured input + Hungarian matching"
echo ""
echo "Input Structure:"
echo "  Position 0: Background (layer 0) - FIXED, low weight"
echo "  Position 1: Target layer - FIXED, high weight (2.0x)"
echo "  Position 2+: Other foreground - SHUFFLED (if enabled)"
echo ""
echo "Input Projection:"
echo "  Structured 6 layers → merged image latent"
echo "  Output = real image latent (PixArt's space!)"
echo ""
echo "Output Projection:"
echo "  Merged latent → 6 layers (any order)"
echo "  Hungarian algorithm finds optimal matching"
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
export NCCL_P2P_DISABLE=1

# Multi-GPU training (2 GPUs)
accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --multi_gpu \
    --mixed_precision=fp16 \
    train_scripts/pretrain_projections.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --merge_weight ${MERGE_WEIGHT} \
    --decompose_weight ${DECOMPOSE_WEIGHT} \
    --background_weight ${BACKGROUND_WEIGHT} \
    --target_layer_weight ${TARGET_LAYER_WEIGHT} \
    ${ENABLE_SHUFFLE}

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
