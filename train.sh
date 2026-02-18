#!/usr/bin/env bash
# ${GPUS:-4}
# set -x


CONFIG=$1
GPUS_ID=${2:-0}    #the default gpu_id is 0 
PORT=${3:-29000}   #the default gpu_id is 1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

GPU_NUM=0

for ((i=0; i<${#GPUS_ID}; i++)); do
    if [[ ${GPUS_ID:i:1} =~ [0-9] ]]; then
        ((GPU_NUM++))
    fi
done

# Use the virtual environment Python
export PYTHON_CMD=/home/bodis/STEERER/.venv/bin/python

echo "export CUDA_VISIBLE_DEVICES=$GPUS_ID"
export CUDA_VISIBLE_DEVICES=${GPUS_ID:-"0"}


# torchrun --nproc_per_node=${GPU_NUM} --master_port ${PORT} tools/train_cc.py --cfg ${CONFIG} 

$PYTHON_CMD -m torch.distributed.launch \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPU_NUM \
    --master_port=$PORT \
    tools/train_cc.py --cfg=$CONFIG --launcher="pytorch"