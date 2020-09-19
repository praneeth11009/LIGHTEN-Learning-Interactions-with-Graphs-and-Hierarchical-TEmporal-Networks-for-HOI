import os
from easydict import EasyDict as edict

cfg = edict()

# GPU to use for training/testing
cfg.GPU = '2'

# Parent directory, for the code base
cfg.PARENT_DIR = '/home/safeer/hoi_graph/LIGHTEN-Learning-Interactions-with-Graphs-and-Hierarchical-TEmporal-Networks-for-HOI/CAD120/'

# path to CAD 120 RGB frames directory
cfg.CAD120_IMAGE_DIR = '/home/safeer/hoi_graph/dataset/All_subjects_images'

# Directory to store/load checkpoints from
cfg.CHECKPOINT_DIR = os.path.join(cfg.PARENT_DIR, 'checkpoints') 

# Data direcotry
cfg.DATA_DIR = os.path.join(cfg.PARENT_DIR, 'data')

# Log directory, to store training logs
cfg.LOG_DIR = os.path.join(cfg.PARENT_DIR, 'logs')

# Spatial subnet variant to use for training
cfg.SPATIAL_SUBNET = 'GCN'

# HOI task, can be detection or anticipation
cfg.TASK = 'detection'
# cfg.TASK = 'anticipation'

# Training parameters
cfg.TRAIN = edict() 

# Indicate whether to train upto frame-level subnet, or the entire model
cfg.TRAIN_LEVEL = 'frame'
# cfg.TRAIN_LEVEL = 'segment'

# Initial learning rate
cfg.TRAIN.LEARNING_RATE = 2e-5

# Batch Size, valid only while frame-level training. During segment-level training (entire model), entire video is a single batch
cfg.TRAIN.BATCH_SIZE = 3

# Total number of epochs
cfg.TRAIN.EPOCHS = 300

# Weight decay
cfg.TRAIN.WEIGHT_DECAY = 0.8

# Step size for reducing the learning rate
cfg.TRAIN.STEPSIZE = 10

# Object loss scaling factor, to account for higher number of objects than humans
cfg.TRAIN.OBJECT_LOSS_SCALE = 2
if cfg.TRAIN_LEVEL == 'segment':
	cfg.TRAIN.OBJECT_LOSS_SCALE = 1

# Dropout for linear layers
cfg.TRAIN.DROPOUT = 0.0