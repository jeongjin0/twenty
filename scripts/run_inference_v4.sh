#!/bin/bash

# Inpainting V4 Inference Script
# Usage: ./run_inference_v4.sh

# ============================================
# Configuration - 수정하세요!
# ============================================
CHECKPOINT="output/inpainting_v4/checkpoints/epoch_10_step_50000.pth"
DATASET_PATH="../data/mulan_coco"  # 데이터셋 경로
IMAGE_INDEX="000000117536"          # 테스트할 이미지 인덱스
OUTPUT_DIR="./inference_results"

# Model paths
T5_PATH="/workspace/twenty/PixArt-alpha"
VAE_PATH="/workspace/twenty/PixArt-alpha/sd-vae-ft-ema"

# Sampling parameters
CFG_SCALE=4.5
STEPS=20
SEED=42
IMAGE_SIZE=256

# ============================================
# Run Inference
# ============================================
mkdir -p ${OUTPUT_DIR}

python scripts/inference_multilayer_noalpha.py \
    --checkpoint ${CHECKPOINT} \
    --dataset_path ${DATASET_PATH} \
    --image_index ${IMAGE_INDEX} \
    --output ${OUTPUT_DIR}/generated_${IMAGE_INDEX}.png \
    --t5_path ${T5_PATH} \
    --vae_path ${VAE_PATH} \
    --cfg_scale ${CFG_SCALE} \
    --steps ${STEPS} \
    --seed ${SEED} \
    --image_size ${IMAGE_SIZE}
