#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
OUT=number_of_predicted_road_pixels.txt
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/infer_calc_number_of_road_pixels.py \
    $CONFIG $CHECKPOINT --out ${OUT} --launcher pytorch ${@:4}
