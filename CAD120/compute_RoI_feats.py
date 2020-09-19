import os
import sys
import pickle as pkl
import numpy as np
import time
import cv2
import math
import torch.nn as nn
import torch
from glob import glob
from skimage.transform import resize

from config import cfg
sys.path.append(cfg.PARENT_DIR+'/models/')
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

# Function to read images of a segment
def read_segment_images(blobs):
    seg_frames_nums = blobs['seg_frames']
    video_id = blobs['video_id']
    frames_dir = [y for x in os.walk(cfg.CAD120_IMAGE_DIR) for y in glob(os.path.join(x[0], str(video_id)))][0]

    start_frame, end_frame = seg_frames_nums[0], seg_frames_nums[1]
    prev_im = np.zeros((480, 640, 3))

    num_frames = 20
    idx = np.arange(start_frame, end_frame)
    stride = int((end_frame-start_frame)/num_frames)
    rem = (end_frame-start_frame)%num_frames
    res_idx = []
    for i in range(num_frames):
      if i < rem :
         res_idx.append(idx[i*(stride+1)])
      else:
         res_idx.append(idx[rem*(stride+1) + (i-rem)*stride - 1])

    images = []
    for num in res_idx:
        img_path = os.path.join(frames_dir, 'RGB_'+str(num)+'.png')
        im_orig  = cv2.imread(img_path)
        if im_orig is None:
            im_orig = prev_im
            print('Image not found, using previous frame')
        else:
            prev_im = im_orig
        im_orig  = im_orig.astype(np.float32)/256.0
        images.append(im_orig)

    images = np.array(images)
    assert len(images) == 20
    return images

# Function to sample 20 frames from a segment, and normalize for ResNet
def crop_pool_norm(batch_images, batch_bboxes, batch_ids):
    batch_size = batch_images.shape[0]
    num_frames = batch_images.shape[1]

    batch_segments = np.zeros([batch_size, 7, num_frames, 224, 224, 3])*1.0
    for b in range(batch_size):
        num_nodes = len(batch_bboxes[b])
        for n in range(num_nodes) :
            for f in range(num_frames) :
                box = batch_bboxes[b, n, f]
                x1, y1, x2, y2 = math.floor(box[0]), math.floor(box[1]), math.floor(box[2]), math.floor(box[3])
                batch_cropped = batch_images[b, f][x1:x2, y1:y2, :]
                if batch_cropped.shape[1] == 0 or batch_cropped.shape[2] == 0:
                    continue
                batch_segments[b, n, f] = resize(batch_cropped, (224, 224, 3))

    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    batch_segments[:, :num_nodes] = (batch_segments[:, :num_nodes] - img_mean)/img_std
    return batch_segments

# Compute global hull bounding box 
def compute_hull_bboxes(batch_bboxes):
    batch_hull_bboxes = []
    for b in range(len(batch_bboxes)):
        hull_bboxes = np.zeros((20, 4))
        for f in range(20):
            x1, y1, x2, y2 = 1e10, 1e10, -1, -1
            for o in range(len(batch_bboxes[b])):
                x1 = min(x1, batch_bboxes[b][o, f, 0])
                y1 = min(y1, batch_bboxes[b][o, f, 1])
                x2 = max(x2, batch_bboxes[b][o, f, 2])
                y2 = max(y2, batch_bboxes[b][o, f, 3])
            hull_bboxes[f, :] = np.array([x1, y1, x2, y2])
        batch_hull_bboxes.append(hull_bboxes)
    return batch_hull_bboxes


training_data_file = os.path.join(cfg.DATA_DIR, 'training_data.p')
testing_data_file = os.path.join(cfg.DATA_DIR, 'testing_data.p')

blobs_segment_train = pkl.load(open(training_data_file, "rb"))
blobs_segment_test = pkl.load(open(testing_data_file , "rb"))

res50 = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
res50 = torch.nn.Sequential(*(list(res50.children())[:-1]))
res50.cuda()
res50.eval()

# Read images and compute ResNet backbone features
def forward_step(mode, i):
    global res50, blobs_segment_test, blobs_segment_train
    if mode == 'train':
        blobs = blobs_segment_train[i]
    elif mode == 'test':
        blobs = blobs_segment_test[i]
    
    batch_images = np.zeros((1, 20, 480, 640, 3)).astype('float')
    for b in range(1):
        batch_images[b, :, :, :, :] = read_segment_images(blobs)

    batch_node_bboxes = [blobs['gt_bboxes']]
    batch_hull_bboxes = compute_hull_bboxes(batch_node_bboxes)
    batch_bboxes = [batch_node_bboxes[0][0], batch_hull_bboxes[0]]
    for o in range(1, len(batch_node_bboxes[0])):
        batch_bboxes.append(batch_node_bboxes[0][o])
    batch_bboxes = np.array([batch_bboxes])
    node_im_segments = crop_pool_norm(batch_images, batch_bboxes, [blobs['video_id']])
    for b in range(1):
        node_im_segments = torch.FloatTensor(node_im_segments[b]).cuda()
        node_im_segments = node_im_segments.permute(0, 1, 4, 2, 3)
        
        res50_feats = np.zeros((7, 20, 2048, 1, 1))
        for o in range(7):
            res50_feats[o, :] = res50(node_im_segments[o, :]).cpu().detach().numpy()

        if mode == 'train':
            blobs_segment_train[i]['res50_feats'] = res50_feats
        elif mode == 'test':
            blobs_segment_test[i]['res50_feats'] = res50_feats

# Process training data
start = time.time()
for i in range(len(blobs_segment_train)):
    forward_step('train', i)
    print('Progress:', str(i+1)+'/'+str(len(blobs_segment_train)), 'time:', time.time()-start)
    start = time.time()

training_data_file = os.path.join(cfg.DATA_DIR, 'training_data_with_RoI_features.p')
pkl.dump(blobs_segment_train, open(training_data_file, 'wb+'))

# Process testing data
for i in range(len(blobs_segment_test)):
    forward_step('test', i)
    print('Progress:', str(i+1)+'/'+str(len(blobs_segment_test)), 'time:', time.time()-start)
    start = time.time()

testing_data_file = os.path.join(cfg.DATA_DIR, 'testing_data_with_RoI_features.p')
pkl.dump(blobs_segment_test, open(testing_data_file, 'wb+'))

  

