#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/workspace/twenty"

nohup python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29501 \
    train_scripts/train_multilayer_ref_noalpha.py \
    --config configs/multilayer_config/PixArt_xl2_img256_multilayer_clip.py \
    --model_type crossattn \
    --work_dir ./output/multilayer_pixart_256_clip \
    > train_clip_256.log 2>&1 &

echo "Training started in background. Check train_clip_256.log for progress."
echo "PID: $!"
