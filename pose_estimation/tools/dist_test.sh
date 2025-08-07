#!/usr/bin/env sh
source /mnt/petrelfs/zhanghao.p/.zhshrc_netnew
conda activate mambapose
cd /mnt/petrelfs/zhanghao.p/zhanghao5201/PoseVMamba/pose_estimation
pwd

PARTITION=$1
JOB_NAME=$2
PORT=$3
CONFIG=$4
CHECKPOINT=$5
GPUS=$6
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="/mnt/petrelfs/zhanghao.p/zhanghao5201/PoseVMamba/pose_estimation/src":$PYTHONPATH 
srun --partition=$PARTITION --time=4-00:00:00 --quotatype=spot \
--mpi=pmi2 \
--gres=gpu:$GPUS \
--job-name=${JOB_NAME} \
--kill-on-bad-exit=1 \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch 

#--async spot

#sh tools/dist_test.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_base_8xb32-210e_coco-384x288.py ../pretrain_model/hrvmamba_base384288.pth 4 77.7
#sh tools/dist_test.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_small_8xb32-210e_coco-384x288.py ../pretrain_model/hrvmamba_small384288.pth 4 76.4

#sh tools/dist_test.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_base_8xb32-210e_coco-256x192.py ../pretrain_model/hrvmamba_base256192.pth 4  76.5
#sh tools/dist_test.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrvmamba_small_8xb32-210e_coco-256x192.py ../pretrain_model/hrvmamba_small256192.pth 4 74.6
#sh tools/dist_test.sh XXXX exp_23 20632 configs/cocofinal/td-hm_hrvmamba_tiny_8xb32-210e_coco-256x192.py ../pretrain_model/hrvmamba_tiny256192.pth 8 69.5

#sh tools/dist_test.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrformer-base_8xb32-210e_coco-384x288.py 2
#sh tools/dist_test.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrformer-small_8xb32-210e_coco-384x288.py 2

#sh tools/dist_test.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrformer-base_8xb32-210e_coco-256x192.py 2
#sh tools/dist_test.sh XXXX exp_21 20338 configs/cocofinal/td-hm_hrformer-small_8xb32-210e_coco-256x192.py 2
#sh tools/dist_test.sh XXXX exp_23 20632 configs/cocofinal/td-hm_hrformer-tiny_8xb32-210e_coco-256x192.py  ../pretrain_model/hrformer_tiny256192.pth 4 68.3


