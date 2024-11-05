#!/bin/bash

CS_PATH='/mnt/data/humanparsing/LIP/'
# CS_PATH='/mnt/data/humanparsing/PPP'
BS=1
GPU_IDS='1'
INPUT_SIZE='473,473' 
# INPUT_SIZE='512,512'
SNAPSHOT_FROM='./snapshots/'
DATASET='val'
NUM_CLASSES=20
# NUM_CLASSES=7

CUDA_VISIBLE_DEVICES=0 python evaluate_multi_LIP.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}