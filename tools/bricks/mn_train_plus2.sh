#!/usr/bin/env bash

set -e
set -x


CONFIG=$1
NODE_NUM=$2
RANK=$3
MASTER_ADDR=$4
PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch  --nnodes=$NODE_NUM --node_rank=$RANK --nproc_per_node=8  --master_addr=$MASTER_ADDR --master_port=$PORT \
    $(dirname "$0")/train_plus2.py $CONFIG --launcher pytorch  --deterministic ${@:5}  
