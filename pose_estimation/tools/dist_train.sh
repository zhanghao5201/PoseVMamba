#!/usr/bin/env bash
source /mnt/petrelfs/zhanghao.p/.zhshrc_netnew
conda activate mambapose

cd /mnt/petrelfs/zhanghao.p/zhanghao5201/PoseVMamba/pose_estimation
pwd

PARTITION=$1
JOB_NAME=$2
PORT=$3
CONFIG=$4
GPUS=$5
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="/mnt/petrelfs/zhanghao.p/zhanghao5201/PoseVMamba/pose_estimation/src":$PYTHONPATH 
srun --partition=$PARTITION --time=5-00:00:00 --quotatype=spot \
--mpi=pmi2 \
--gres=gpu:$GPUS \
--job-name=${JOB_NAME} \
--kill-on-bad-exit=1 \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py \
    $CONFIG --launcher pytorch --resume=auto


###
#sh tools/dist_train.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_base_8xb32-210e_coco-384x288.py 2
#sh tools/dist_train.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_small_8xb32-210e_coco-384x288.py 2

#sh tools/dist_train.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_base_8xb32-210e_coco-256x192.py 2
#sh tools/dist_train.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_small_8xb32-210e_coco-256x192.py 2
#sh tools/dist_train.sh XXXX exp_23 20632 configs/cocofinal/td-hm_hrvmamba_tiny_8xb32-210e_coco-256x192.py 2

#sh tools/dist_train.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrformer-base_8xb32-210e_coco-384x288.py 2
#sh tools/dist_train.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrformer-small_8xb32-210e_coco-384x288.py 2

#sh tools/dist_train.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrformer-base_8xb32-210e_coco-256x192.py 2
#sh tools/dist_train.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrformer-small_8xb32-210e_coco-256x192.py 2
#sh tools/dist_train.sh XXXX exp_23 20632 configs/cocofinal/td-hm_hrformer-tiny_8xb32-210e_coco-256x192.py 2


