#!/usr/bin/env bash
source ~/.zhshrc_net
conda activate mambapose



PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=$4
PORT=$5
CPUS_PER_TASK=${CPUS_PER_TASK:-10}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=spot --async \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch 
#--async 
# sh tools/dist_train.sh XXX exp 'configs/hrvmamba/hrvmamba_base_160k_ade20k-512x512_base.py' 8 27131
