#!/bin/bash
# Train MultiLayer Reference Model with CLIP Encoder and Text Dropout
# 256x256 resolution, CrossAttention architecture

export PYTHONPATH="${PYTHONPATH}:/workspace/twenty"
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 사용할 GPU 지정

# Configuration
CONFIG="configs/multilayer_config/PixArt_xl2_img256_multilayer_clip.py"
WORK_DIR="./output/multilayer_pixart_256_clip_$(date +%Y%m%d_%H%M%S)"
NUM_GPUS=4

# Model settings
MODEL_TYPE="crossattn"  # Use cross-attention architecture
USE_REF=true

# Training settings
BATCH_SIZE=8  # per GPU
GRAD_ACCUM=2
NUM_EPOCHS=100

# Resume from checkpoint (optional)
# RESUME_FROM="./output/multilayer_pixart_256_clip/checkpoint-10000.pth"

echo "=================================================="
echo "Training MultiLayer PixArt with CLIP & Text Dropout"
echo "=================================================="
echo "Config: $CONFIG"
echo "Work dir: $WORK_DIR"
echo "GPUs: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE x $GRAD_ACCUM x $NUM_GPUS = $(($BATCH_SIZE * $GRAD_ACCUM * $NUM_GPUS))"
echo "Model type: $MODEL_TYPE"
echo "Use CLIP Ref Encoder: True"
echo "Text Dropout Prob: 0.5"
echo "=================================================="

# Run training with accelerate
accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=fp16 \
    --multi_gpu \
    train_scripts/train_multilayer_ref_noalpha.py \
    --config=$CONFIG \
    --model_type=$MODEL_TYPE \
    --work_dir=$WORK_DIR \
    --train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$GRAD_ACCUM \
    --num_epochs=$NUM_EPOCHS \
    ${RESUME_FROM:+--resume_from=$RESUME_FROM}

echo "=================================================="
echo "Training completed!"
echo "Output saved to: $WORK_DIR"
echo "=================================================="
