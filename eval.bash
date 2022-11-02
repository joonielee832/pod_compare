#!/bin/bash

python3 src/apply_net.py \
--dataset-dir /home/data/bdd100k \
--test-dataset bdd_val \
--config-file BDD-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var.yaml \
--inference-config Inference/bayes_od.yaml