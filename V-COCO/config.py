import os
from easydict import EasyDict as edict

cfg = edict()

# GPU number 
cfg.GPU = '2'

# Parent directory, for the code base
cfg.PARENT_DIR = '/home/safeer/hoi_graph/LIGHTEN-Learning-Interactions-with-Graphs-and-hierarchical-TEmporal-Networks-for-HOI/V-COCO/'

# path to V-COCO image directory
cfg.VCOCO_IMAGE_DIR = '/home/safeer/hoi_graph/dataset/v-coco/images/traintest2017/'

# path to V-COCO data
cfg.VCOCO_DIR = '/home/safeer/hoi_graph/dataset/v-coco/'

# path to V-COCO eval data
cfg.VCOCO_EVAL_DIR = '/home/safeer/hoi_graph/dataset/vcocoeval/'

# Directory to store/load checkpoints from
cfg.CHECKPOINT_DIR = os.path.join(cfg.PARENT_DIR, 'checkpoints') 

# Data direcotry
cfg.DATA_DIR = os.path.join(cfg.PARENT_DIR, 'data')

# Log directory, to store training logs
cfg.LOG_DIR = os.path.join(cfg.PARENT_DIR, 'logs')

# Path to COCO API
cfg.COCO_PATH = '/home/safeer/v-coco'

# Spatial subnet variant to use for training
cfg.SPATIAL_SUBNET = 'GCN'

# Whether to use validation data for training
cfg.TRAIN_VAL = True

# Training parameters
cfg.TRAIN = edict()

# Initial learning rate
cfg.TRAIN.LEARNING_RATE = 2e-4

# Batch size
cfg.TRAIN.BATCHSIZE = 10

# Total epochs to train for
cfg.TRAIN.EPOCHS = 300

# Factor for reducing the learning rate
cfg.TRAIN.WEIGHT_DECAY = 0.8

# Step size for reducing the learning rate, currently support each batch
cfg.TRAIN.STEPSIZE = 10

# Dropout 
cfg.TRAIN.DROPOUT = 0.0

## Testing params
cfg.TEST = edict()

# Human threshold
cfg.TEST.HUMAN_THRES = 0.8

# Object threshold 
cfg.TEST.OBJECT_THRES = 0.4

# Prior flag
cfg.TEST.PRIOR_FLAG = 3
