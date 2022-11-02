#!/bin/bash

python3 src/visualize_predictions.py \
--num-gpus 2 \
--dataset-dir /home/data/bdd100k \
--config-file BDD-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var.yaml \
--test-dataset bdd_val \
--inference-config Inference/bayes_od.yaml