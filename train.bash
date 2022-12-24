#!/bin/bash

NUM_GPUS=$1

python3 src/train_net.py \
--num-gpus $NUM_GPUS \
--dataset-dir /home/data/bdd100k \
--config-file BDD-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var.yaml \
--random-seed 0 \
--resume