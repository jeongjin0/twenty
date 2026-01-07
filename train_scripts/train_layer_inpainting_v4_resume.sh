#!/bin/bash

# Layer-wise Inpainting v4 Training - RESUME
# Resume from a previous checkpoint
#
# Usage:
#   1. Auto-detect latest checkpoint (recommended):
#      bash train_scripts/train_layer_inpainting_v4_resume.sh
#
#   2. Specify checkpoint manually:
#      Edit RESUME_CHECKPOINT variable below, then run:
#      bash train_scripts/train_layer_inpainting_v4_resume.sh
#
# Note: This script automatically finds the most recent checkpoint
#       based on file modification time.

CONFIG="configs/layer_inpainting_config.py"
OUTPUT_DIR="output/inpainting_v4"
PRETRAINED_PIXART="PixArt-alpha/PixArt-XL-2-256x256.pth"  # Update this path
PRETRAINED_PROJECTIONS="output/projection_pretrain_v3/checkpoints/epoch_05.pth"  # From projection pretraining

# Resume checkpoint (auto-detect latest or specify manually)
RESUME_CHECKPOINT=""  # Leave empty to auto-detect latest checkpoint

echo "============================================"
echo "Layer-wise Inpainting v4 Training - RESUME"
echo "============================================"
echo ""

# Auto-detect latest checkpoint if not specified
if [ -z "$RESUME_CHECKPOINT" ]; then
    CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"

    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "ERROR: Checkpoint directory not found: ${CHECKPOINT_DIR}"
        echo "Please run initial training first or specify RESUME_CHECKPOINT manually."
        exit 1
    fi

    # Find all checkpoint files and sort by modification time (most recent first)
    LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "*.pth" -type f -printf '%T@ %p\n' | \
                        sort -rn | head -1 | cut -d' ' -f2-)

    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found in ${CHECKPOINT_DIR}"
        echo "Please run initial training first."
        exit 1
    fi

    RESUME_CHECKPOINT="$LATEST_CHECKPOINT"
    CHECKPOINT_NAME=$(basename "$RESUME_CHECKPOINT")
    echo "Auto-detected latest checkpoint: ${CHECKPOINT_NAME}"
    echo "  Full path: ${RESUME_CHECKPOINT}"
else
    echo "Using specified checkpoint: ${RESUME_CHECKPOINT}"
fi

# Verify checkpoint exists
if [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint file not found: ${RESUME_CHECKPOINT}"
    exit 1
fi

echo ""
echo "Resume Configuration:"
echo "  Config: ${CONFIG}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Resume from: ${RESUME_CHECKPOINT}"
echo ""
echo "Note: Training will continue from the saved epoch/step."
echo "      Optimizer and LR scheduler states will be restored."
echo ""
echo "Multi-GPU: 2 GPUs, mixed precision FP16"
echo "============================================"
echo ""

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export NCCL_P2P_DISABLE=1

# Multi-GPU training (2 GPUs) with resume
accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --multi_gpu \
    train_scripts/train_layer_inpainting.py \
    --config ${CONFIG} \
    --pretrained_pixart ${PRETRAINED_PIXART} \
    --pretrained_projections ${PRETRAINED_PROJECTIONS} \
    --resume ${RESUME_CHECKPOINT}

echo ""
echo "============================================"
echo "Training resumed and completed!"
echo "============================================"
echo "Checkpoints saved to: ${OUTPUT_DIR}/checkpoints/"
echo "============================================"
