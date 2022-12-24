from collections import ChainMap

# Detectron imports
from detectron2.data import MetadataCatalog


# Construct BDD metadata
BDD_THING_CLASSES = ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']
BDD_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i + 1: i} for i in range(len(BDD_THING_CLASSES))]))

# Construct KITTI metadata
KITTI_THING_CLASSES = ['car', 'person'] # , 'bike'
KITTI_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(
    ChainMap(*[{i + 1: i} for i in range(len(KITTI_THING_CLASSES))]))

# MAP BDD to KITTI contiguous id to be used for inference on KITTI for models
# trained on BDD.
# BDD_TO_KITTI_CONTIGUOUS_ID = dict(ChainMap(
#     *[{BDD_THING_CLASSES.index(kitti_thing_class): KITTI_THING_CLASSES.index(kitti_thing_class)} for kitti_thing_class in
#       KITTI_THING_CLASSES]))
