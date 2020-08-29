import time
import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import numpy as np
import math
import cv2
from skimage.transform import resize
import sklearn.metrics

from config import cfg
random.seed(0)
sys.path.append(os.path.join(cfg.PARENT_DIR, 'models'))
from model_VCOCO_LIGHTEN import LIGHTEN_image
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

print('==> Create model and initialize')
experiment_name = cfg.SPATIAL_SUBNET
if cfg.TRAIN_VAL :
    experiment_name += '_trainVal'
print('Running experiment', experiment_name)

model = LIGHTEN_image(spatial_subnet=cfg.SPATIAL_SUBNET, drop=cfg.TRAIN.DROPOUT)
model.float().cuda()

checkpoint_file = os.path.join(cfg.CHECKPOINT_DIR, 'checkpoint_'+experiment_name+'.pth')
if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded model from checkpoint')
else:
    print('Checkpoint file does not exist')
    exit(1)

training_data_file = os.path.join(cfg.DATA_DIR, 'training_data_with_RoI_features.p')
testing_data_file = os.path.join(cfg.DATA_DIR, 'testing_data_with_RoI_features.p')

blobs_train = pkl.load(open(training_data_file, "rb"))
blobs_test = pkl.load(open(testing_data_file , "rb"))

def sigmoid_cross_entropy_loss(logits, labels, mask):
    logits, labels, mask = logits[0], labels[0], mask[0]
    loss_1 = logits[logits>=0] - logits[logits>=0]*labels[logits>=0] + torch.log(1 + torch.exp(-logits[logits>=0]))
    loss_1 = loss_1*mask[logits>=0]
    loss_2 = - logits[logits<0]*labels[logits<0] + torch.log(1 + torch.exp(logits[logits<0]))
    loss_2 = loss_2*mask[logits<0]
    loss = torch.mean(torch.cat((loss_1, loss_2), 0))
    return loss

def forward_step(batch_blobs):
    label_HO = torch.Tensor([blob['gt_class_HO'] for blob in batch_blobs]).long().cuda().squeeze()
    mask_HO = torch.Tensor([blob['Mask_HO'] for blob in batch_blobs]).squeeze().cuda()

    batch_human_boxes = torch.FloatTensor(np.array([blob['H_boxes'] for blob in batch_blobs])).cuda()
    batch_object_boxes = torch.FloatTensor(np.array([blob['O_boxes'] for blob in batch_blobs])).cuda()

    batch_res_human_feats = torch.zeros((cfg.TRAIN.BATCHSIZE, 2048, 1, 1)).float().cuda()
    batch_res_object_feats = torch.zeros((cfg.TRAIN.BATCHSIZE, 2048, 1, 1)).float().cuda()
    for b in range(cfg.TRAIN.BATCHSIZE):
        res_feats = batch_blobs[b]['res50_feats']
        batch_res_human_feats[b, :, :, :] = res_feats[0]
        batch_res_object_feats[b, :, :, :] = res_feats[1]

    action_cls_scores = model.forward(batch_human_boxes, batch_object_boxes, batch_res_human_feats, batch_res_object_feats)
    action_loss = sigmoid_cross_entropy_loss(action_cls_scores, label_HO, mask_HO)
    
    batch_action_loss = action_loss.item()

    action_cls_scores, label_HO = action_cls_scores.cpu().detach().numpy(), label_HO.cpu().detach().numpy()
    return batch_action_loss, action_cls_scores, label_HO

print('Start validation')

epoch_start_time = time.time()

total_action_loss = 0.0
HO_gts, HO_scores = np.zeros([len(blobs_train), 26]), np.zeros([len(blobs_train), 26])*1.0

num_batches = int(len(blobs_train)/cfg.TRAIN.BATCHSIZE)

start = time.time()
for i in range(num_batches):
    if (i+1)%100 == 0:
        print('Progress:',str(i+1)+'/'+str(num_batches), 'time:', time.time()-start)
        start = time.time()
    batch_blobs = blobs_train[cfg.TRAIN.BATCHSIZE*i : cfg.TRAIN.BATCHSIZE*(i+1)]

    batch_action_loss, action_scores, action_gts = forward_step(batch_blobs)
    
    total_action_loss += batch_action_loss

    HO_scores[cfg.TRAIN.BATCHSIZE*i : cfg.TRAIN.BATCHSIZE*(i+1)] = action_scores
    HO_gts[cfg.TRAIN.BATCHSIZE*i : cfg.TRAIN.BATCHSIZE*(i+1)] = action_gts 

total_action_loss = total_action_loss / num_batches

HO_gts_new, HO_scores_new = [], []
for i in range(HO_gts.shape[1]):
    if np.sum(HO_gts[:, i]) != 0.0:
        HO_gts_new.append(HO_gts[:,i])
        HO_scores_new.append(HO_scores[:, i])
HO_gts = np.transpose(np.array(HO_gts_new))
HO_scores = np.transpose(np.array(HO_scores_new))

action_mAP_macro = sklearn.metrics.average_precision_score(y_true=HO_gts, y_score=HO_scores, average='macro')*100

print('Action_loss: %.6f, action_mAP: %.4f, time: %.3f s/iter' %  
      (total_action_loss, action_mAP_macro, time.time()-epoch_start_time))

# Testing 
epoch_start_time = time.time()
total_action_loss = 0.0
HO_gts, HO_scores = np.zeros([len(blobs_test), 26]), np.zeros([len(blobs_test), 26])*1.0 

num_batches = int(len(blobs_test)/cfg.TRAIN.BATCHSIZE)
for i in range(num_batches):
    batch_blobs = blobs_test[cfg.TRAIN.BATCHSIZE*i : cfg.TRAIN.BATCHSIZE*(i+1)]
    batch_action_loss, action_scores, action_gts = forward_step(batch_blobs)
    
    total_action_loss += batch_action_loss

    HO_scores[cfg.TRAIN.BATCHSIZE*i : cfg.TRAIN.BATCHSIZE*(i+1)] = action_scores
    HO_gts[cfg.TRAIN.BATCHSIZE*i : cfg.TRAIN.BATCHSIZE*(i+1)] = action_gts

total_action_loss = total_action_loss / num_batches

HO_gts_new, HO_scores_new = [], []
for i in range(HO_gts.shape[1]):
    if np.sum(HO_gts[:, i]) != 0.0:
        HO_gts_new.append(HO_gts[:,i])
        HO_scores_new.append(HO_scores[:, i])
HO_gts = np.transpose(np.array(HO_gts_new))
HO_scores = np.transpose(np.array(HO_scores_new))

action_mAP_macro = sklearn.metrics.average_precision_score(y_true=HO_gts, y_score=HO_scores, average='macro')*100
   
print('Test, action_loss: %.6f, action_mAP: %.4f, time: %.3f s/iter' %  
      (total_action_loss, action_mAP_macro, time.time()-epoch_start_time))
print('\n')

