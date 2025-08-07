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
RESUME=$6
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
    --local-rank 0 --eval \
    --batch-size $BATCH_SIZE \
    --data-path /mnt/petrelfs/share/images \
    --output work_dirs --launcher="slurm" --resume $RESUME


# sh dist_test.sh XXXX base 'configs/hrvmamba/hrvmamba_base.yaml' 4 128 PoseVMamba/pretrain_model/hrvmamba_base.pth * Acc@1 84.164
# sh dist_test.sh XXXX nano 'configs/hrvmamba/hrvmamba_nano.yaml' 4 256 PoseVMamba/pretrain_model/hrvmamba_nano.pth * Acc@1 74.770
# sh dist_test.sh XXXX small 'configs/hrvmamba/hrvmamba_small.yaml' 4 256 PoseVMamba/pretrain_model/hrvmamba_small.pth * Acc@1 81.302
# sh dist_test.sh XXXX tiny 'configs/hrvmamba/hrvmamba_tiny.yaml' 4 256 PoseVMamba/pretrain_model/hrvmamba_tiny.pth  * Acc@1 78.572

# sh dist_test.sh XXXX base 'configs/hrvmamba/hrformer_base.yaml' 4 128 PoseVMamba/pretrain_model/hrformer_base_best.pth 83.3%
# sh dist_test.sh XXXX nano 'configs/hrvmamba/hrformer_nano.yaml' 4 256 PoseVMamba/pretrain_model/hrformer_nano_best.pth 74.3%
# sh dist_test.sh XXXX small 'configs/hrvmamba/hrformer_small.yaml' 8 256 PoseVMamba/pretrain_model/hrformer_small_best.pth 80.8%
# sh dist_test.sh XXXX tiny 'configs/hrvmamba/hrformer_tiny.yaml' 4 256 PoseVMamba/pretrain_model/hrformer_tiny_best.pth   77.6%

