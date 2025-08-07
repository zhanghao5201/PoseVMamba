#!/usr/bin/env bash

source ~/.zhshrc_netnew
conda activate PoseVMamba

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=$4
PORT=$5
CHECKPOINT=$6
CPUS_PER_TASK=${CPUS_PER_TASK:-10}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=spot --async \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch --tta

# sh tools/dist_test.sh XXX exp 'configs/hrvmamba/hrvmamba_base_160k_ade20k-512x512_base.py' 8 17234 hrvmamba_base_160k_ade20k-512x512_base/iter_160000.pth
