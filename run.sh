#!/bin/bash
uname -a
#date
#env
date
# CS_PATH='/mnt/data/humanparsing/ATR'
CS_PATH='/mnt/data/humanparsing/LIP/'
# CS_PATH='/mnt/data/humanparsing/PPP'
#LR=2.0e-3    #[59.85]
LR=2.5e-3    #[60.106]
# LR=3.0e-3      #[59.87]
# LR=3.0e-2
WD=5e-4
# WD=1e-4
BS=4    #from 4 to 6 2023/4/2
GPU_IDS=0,1,2,3
RESTORE_FROM='./dataset/resnet101-imagenet.pth'
# RESTORE_FROM='./dataset/van_base_828.pth.tar'
INPUT_SIZE='473,473'  
# INPUT_SIZE='512,512' 
SNAPSHOT_DIR='./snapshots'
DATASET='train'
NUM_CLASSES=20 
# NUM_CLASSES=7
EPOCHS=150

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

    python -m torch.distributed.launch --nproc_per_node=4 --nnode=1 \
    --node_rank=0 --master_addr=172.130.229.210 --master_port 29500 train.py \
       --data-dir ${CS_PATH} \
       --random-mirror\
       --random-scale\
       --restore-from ${RESTORE_FROM}\
       --gpu ${GPU_IDS}\
       --learning-rate ${LR}\
       --weight-decay ${WD}\
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --snapshot-dir ${SNAPSHOT_DIR}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES} \
       --epochs ${EPOCHS}
