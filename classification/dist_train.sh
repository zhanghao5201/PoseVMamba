#!/usr/bin/env bash

source /mnt/petrelfs/zhanghao.p/.zhshrc_netnew
conda activate PoseVMamba

cd PoseVMamba/classification
pwd


PARTITION=$1
JOB_NAME=$2
CONFIG=$3
WORK_DIR=$4
GPUS=$4
GPUS_PER_NODE=$4
BATCH_SIZE=$5
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="PoseVMamba/classification/src":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=spot \
    ${SRUN_ARGS} \
python -u main.py \
    --cfg ${CONFIG} \
    --accumulation-steps 1 \
    --local-rank 0 \
    --batch-size $BATCH_SIZE \
    --data-path /mnt/petrelfs/share/images \
    --output work_dirs --launcher="slurm" 

# sh dist_train.sh XXXX small 'configs/hrvmamba/hrvmamba_small.yaml' 8 128
# sh dist_train.sh XXXX tiny 'configs/hrvmamba/hrvmamba_tiny.yaml' 8 256 
# sh dist_train.sh XXXX nano 'configs/hrvmamba/hrvmamba_nano.yaml' 8 256 
# sh dist_train.sh XXXX base 'configs/hrvmamba/hrvmamba_base.yaml' 8 256