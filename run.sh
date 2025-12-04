#!/bin/bash

# 设置NCCL环境变量
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TIMEOUT=3600

# 执行训练命令
torchrun \
    --nproc_per_node=2 SAM-Lung/train_multi_GPU_Mona.py \
    --data-path data/ \
    --device cuda \
    --num-classes 1 \
    --batch-size 1 \
    --epochs 60 \
    --lr 0.01 \
    --output-dir ./multi_train \
    --sync_bn True \
    --workers 16 \
    --save-best True \
    --world-size 2 \
    --amp True 
    # --resume multi_train/best_model.pth