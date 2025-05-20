#!/bin/bash

# Environment Variables
export KEY_HF="xxxx"
export CUDA_LAUNCH_BLOCKING=0
export TMPDIR="./tmp"
export TOOLKIT_DIR="."
export PYTHONPATH="./toolkitmoe/moe_pretrain_model":$PYTHONPATH

# GPU Configuration
export CUDA_VISIBLE_DEVICES="4,5,6,7"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# Calculate number of GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Distributed Training Configuration
MASTER_ADDR="127.0.0.1"  # Master node address (change as needed)
MASTER_PORT=12922        # Master node port
export MASTER_PORT=$MASTER_PORT

# NCCL Configuration for Distributed Training
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=23
export NCCL_SOCKET_TIMEOUT=3600

# Change to model directory
cd ./moe_pretrain_model

while true; do
    echo "Starting stage sft"
    python -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        ./toolkitmoe/moe_pretrain_model/run.py \
        ./toolkitmoe/moe_pretrain_model/sweeps/slimpajama_moe_no_attmoe_154M.yaml

    if [ $? -eq 0 ]; then
        echo "Training completed successfully!"
        break
    else
        echo "Training failed. Restarting..."
        sleep 3
    fi
done
