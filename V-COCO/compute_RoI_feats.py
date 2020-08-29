import time
import os
import sys
import random
import torch
import torch.nn as nn
import pickle as pkl
import numpy as np
import math
import cv2
from skimage.transform import resize

from config import cfg

def crop_pool_norm(batch_images, batch_boxes):
    batch_segments = np.zeros([len(batch_images), 224, 224, 3])*1.0

    st = time.time()
    for i in range(len(batch_images)):
        box = batch_boxes[i]
        x1, y1, x2, y2 = math.floor(box[0]), math.floor(box[1]), math.floor(box[2]), math.floor(box[3])
        xmax = batch_images[i].shape[1]
        ymax = batch_images[i].shape[0]
        assert x2 <= xmax
        assert y2 <= ymax
        batch_cropped = batch_images[i][y1:y2, x1:x2, :]

        batch_segments[i] = resize(batch_cropped, (224, 224, 3))

    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    batch_segments = (batch_segments - img_mean)/img_std
    return batch_segments

def preprocess(blobs):
    for i in range(len(blobs)):
        blob = blobs[i]

        human_boxes = blob['H_boxes'].reshape(blob['H_boxes'].shape[1])
        object_boxes = blob['O_boxes'].reshape(blob['O_boxes'].shape[1])
        human_boxes, object_boxes = human_boxes[1:], object_boxes[1:]

        blobs[i]['H_boxes'] = human_boxes
        blobs[i]['O_boxes'] = object_boxes
    return blobs

start = time.time()

training_data_file = os.path.join(cfg.DATA_DIR, 'training_data.p')
testing_data_file = os.path.join(cfg.DATA_DIR, 'testing_data.p')

blobs_train = pkl.load(open(training_data_file, "rb"))
blobs_test = pkl.load(open(testing_data_file , "rb"))

blobs_train = preprocess(blobs_train)
blobs_test = preprocess(blobs_test)

res50 = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
res50 = torch.nn.Sequential(*(list(res50.children())[:-1]))
res50.cuda()
res50.eval()

def forward_step(i, is_train):
    global blobs_train, blobs_test
    if is_train:
        blob = blobs_train[i]
    else:
        blob = blobs_test[i]

    image_id = blob['image_id']
    im_file = os.path.join(cfg.VCOCO_IMAGE_DIR, (str(image_id)).zfill(12) + '.jpg' )
    im = cv2.imread(im_file)
    im_orig = im.astype(np.float32, copy=True)/256.0

    human_bbox = blob['H_boxes']
    object_bbox = blob['O_boxes']

    segments = torch.FloatTensor(crop_pool_norm(np.array([im_orig, im_orig]),\
                    np.array([human_bbox, object_bbox]))).cuda().permute(0, 3, 1, 2)

    res50_feat = res50(segments).detach().cpu()

    if is_train : 
        blobs_train[i]['res50_feats'] = res50_feat
    else :
        blobs_test[i]['res50_feats'] = res50_feat

print('Process train data')
num_batches = len(blobs_train)
start = time.time()
for i in range(num_batches):
    if (i+1)%100 == 0:
        print('Progress:', str(i)+'/'+str(num_batches), 'Time:', time.time()-start); start = time.time()
    forward_step(i, True)

training_data_file = os.path.join(cfg.DATA_DIR, 'training_data_with_RoI_features.p')
pkl.dump(blobs_train, open(training_data_file, 'wb+'))

print('Process test data')
num_batches = len(blobs_test)
start = time.time()
for i in range(num_batches):
    if (i+1)%100 == 0:
        print('Progress:', str(i)+'/'+str(num_batches), 'Time:', time.time()-start); start = time.time()
    forward_step(i, False)

testing_data_file = os.path.join(cfg.DATA_DIR, 'testing_data_with_RoI_features.p')
pkl.dump(blobs_test, open(testing_data_file, 'wb+'))