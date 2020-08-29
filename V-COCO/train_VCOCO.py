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
learning_rate = cfg.TRAIN.LEARNING_RATE

experiment_name = cfg.SPATIAL_SUBNET
if cfg.TRAIN_VAL :
    experiment_name += '_trainVal'
print('Running experiment', experiment_name)

if not os.path.exists(cfg.LOG_DIR):
    os.system('mkdir '+cfg.LOG_DIR)
log_file = open(os.path.join(cfg.LOG_DIR, 'log_'+experiment_name+'.txt'), "a+")

model = LIGHTEN_image(spatial_subnet=cfg.SPATIAL_SUBNET, drop=cfg.TRAIN.DROPOUT)
model.float().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def save_checkpoint(state):
    torch.save(state, os.path.join(cfg.CHECKPOINT_DIR, 'checkpoint_'+experiment_name+'.pth'))

training_data_file = os.path.join(cfg.DATA_DIR, 'training_data_with_RoI_features.p')
testing_data_file = os.path.join(cfg.DATA_DIR, 'testing_data_with_RoI_features.p')

blobs_train = pkl.load(open(training_data_file, "rb"))
blobs_test = pkl.load(open(testing_data_file , "rb"))

if cfg.TRAIN_VAL :
    blobs_train += blobs_test

def sigmoid_cross_entropy_loss(logits, labels, mask):
    logits, labels, mask = logits[0], labels[0], mask[0]
    loss_1 = logits[logits>=0] - logits[logits>=0]*labels[logits>=0] + torch.log(1 + torch.exp(-logits[logits>=0]))
    loss_1 = loss_1*mask[logits>=0]
    loss_2 = - logits[logits<0]*labels[logits<0] + torch.log(1 + torch.exp(logits[logits<0]))
    loss_2 = loss_2*mask[logits<0]
    loss = torch.mean(torch.cat((loss_1, loss_2), 0))
    return loss

def forward_step(batch_blobs, is_train):
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

    if is_train:
        optimizer.zero_grad()
        action_loss.backward()
        optimizer.step()
    
    batch_action_loss = action_loss.item()

    action_cls_scores, label_HO = action_cls_scores.cpu().detach().numpy(), label_HO.cpu().detach().numpy()
    return batch_action_loss, action_cls_scores, label_HO

print('Start training')
mAP_save = [] 
mAP_save.append(0)

for epoch in range(cfg.TRAIN.EPOCHS): 
    log_file = open(os.path.join(cfg.LOG_DIR, 'log_'+experiment_name+'.txt'), "a+")

    epoch_start_time = time.time()
    random.shuffle(blobs_train)

    total_action_loss = 0.0
    HO_gts, HO_scores = np.zeros([len(blobs_train), 26]), np.zeros([len(blobs_train), 26])*1.0

    num_batches = int(len(blobs_train)/cfg.TRAIN.BATCHSIZE)

    start = time.time()
    for i in range(num_batches):
        if (i+1)%100 == 0:
            print('Progress:',str(i+1)+'/'+str(num_batches), 'time:', time.time()-start)
            start = time.time()
        batch_blobs = blobs_train[cfg.TRAIN.BATCHSIZE*i : cfg.TRAIN.BATCHSIZE*(i+1)]

        batch_action_loss, action_scores, action_gts = forward_step(batch_blobs, True)
        
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
   
    print('Epoch:%02d, action_loss: %.6f, action_mAP: %.4f, lr: %.8f, time: %.3f s/iter' %  
          (epoch, total_action_loss, action_mAP_macro, learning_rate, time.time()-epoch_start_time))
    log_file.write('Epoch:%02d, action_loss: %.6f, action_mAP: %.4f lr: %.8f, time: %.3f s/iter\n' %  
          (epoch, total_action_loss, action_mAP_macro, learning_rate, time.time()-epoch_start_time)) 

    # Testing 
    epoch_start_time = time.time()
    total_action_loss = 0.0
    HO_gts, HO_scores = np.zeros([len(blobs_test), 26]), np.zeros([len(blobs_test), 26])*1.0 

    num_batches = int(len(blobs_test)/cfg.TRAIN.BATCHSIZE)
    for i in range(num_batches):
        batch_blobs = blobs_test[cfg.TRAIN.BATCHSIZE*i : cfg.TRAIN.BATCHSIZE*(i+1)]
        batch_action_loss, action_scores, action_gts = forward_step(batch_blobs, False)
        
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
    log_file.write('Test, action_loss: %.6f, action_mAP: %.4f, time: %.3f s/iter\n' %  
          (total_action_loss, action_mAP_macro, time.time()-epoch_start_time))

    if (epoch+1)%cfg.TRAIN.STEPSIZE == 0:
        if learning_rate > 1e-7:
            learning_rate *= cfg.TRAIN.WEIGHT_DECAY
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            print('decrease lr to', learning_rate)

    if action_mAP_macro > mAP_save[-1] and epoch >= 0 :
        mAP_save.append(action_mAP_macro)
        print('Save checkpoint')
        log_file.write('Save checkpoint')
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'learning_rate': learning_rate,
            'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
            })

    print('\n')
    log_file.write('\n\n')
    log_file.close()

