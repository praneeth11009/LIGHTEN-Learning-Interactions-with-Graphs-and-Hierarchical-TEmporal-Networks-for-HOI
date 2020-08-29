import time
import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import numpy as np
import cv2
import math
from skimage.transform import resize
import sklearn.metrics

from config import cfg
sys.path.append(os.path.join(cfg.PARENT_DIR, 'models/'))
from model_CAD120_LIGHTEN import LIGHTEN_frame_level, LIGHTEN_segment_level

random.seed(0)
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

experiment_name = cfg.SPATIAL_SUBNET+'_'+cfg.TRAIN_LEVEL+'_'+cfg.TASK
print('Running experiment', experiment_name)

if not os.path.exists(cfg.LOG_DIR):
    os.system('mkdir '+cfg.LOG_DIR)
log_file = open(os.path.join(cfg.LOG_DIR, 'log_'+experiment_name+'.txt'), "a+")

print('Initialize model')
if cfg.TRAIN_LEVEL == 'frame':
    model = LIGHTEN_frame_level(spatial_subnet=cfg.SPATIAL_SUBNET, dropout=cfg.TRAIN.DROPOUT)
elif cfg.TRAIN_LEVEL == 'segment':
    model = LIGHTEN_segment_level(spatial_subnet=cfg.SPATIAL_SUBNET, dropout=cfg.TRAIN.DROPOUT)

model.float().cuda()
criterion = nn.CrossEntropyLoss()
learning_rate = cfg.TRAIN.LEARNING_RATE
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def save_checkpoint(state):
    torch.save(state, os.path.join(cfg.CHECKPOINT_DIR, 'checkpoint_'+experiment_name+'.pth'))

# Load annotated data from files
training_data_file = os.path.join(cfg.DATA_DIR, 'training_data_with_RoI_features.p')
testing_data_file = os.path.join(cfg.DATA_DIR, 'testing_data_with_RoI_features.p')

blobs_segment_train = pkl.load(open(training_data_file, "rb"))
blobs_segment_test = pkl.load(open(testing_data_file , "rb"))

# Function which does a forward pass (back-prop too for training data), on a single batch
def forward_step(batch_blobs, is_train):
    batchSize = len(batch_blobs)
    if batchSize == 0:
        return 0, 0, 0, [], [], [], []
    batch_loss, batch_human_loss, batch_object_loss = 0.0, 0.0, 0.0

    label_H = []
    for b in range(batchSize):
        label_H.append(batch_blobs[b]['label_H'])
    label_H = np.array(label_H)

    label_O = []
    for b in range(batchSize):
        label_O.append([])
        for j in range(len(batch_blobs[b]['label_O'])):
            label_O[b].append(batch_blobs[b]['label_O'][j])

    num_objs = [len(x) for x in label_O]

    one_hot_encodings = np.zeros([batchSize, 5, 20, 10])
    object_one_hot_vectors = [x['object_one_hot_vectors'] for x in batch_blobs]

    for b in range(batchSize):
        for i in range(num_objs[b]):
            for f in range(20):
                one_hot_encodings[b, :num_objs[b], f, :] = object_one_hot_vectors[b][:]

    batch_bboxes = [blob['gt_bboxes'] for blob in batch_blobs]
    
    batch_node_bboxes = np.zeros([batchSize, 6, 20, 4]).astype('float')
    for b in range(batchSize):
        n_nodes = len(batch_bboxes[b])
        batch_node_bboxes[b, :n_nodes, :, ::2] = batch_bboxes[b][:, :, ::2]/480.0
        batch_node_bboxes[b, :n_nodes, :, 1::2] = batch_bboxes[b][:, :, 1::2]/640.0

    one_hot_encodings = torch.FloatTensor(one_hot_encodings).cuda()
    batch_node_bboxes = torch.FloatTensor(batch_node_bboxes).cuda()

    if cfg.SPATIAL_SUBNET == 'GCN':
    	res_features = np.zeros((batchSize, 6, 20, 2048, 1, 1)).astype('float')
    	for b in range(batchSize):
    		res_features[b, 0] = batch_blobs[b]['res50_feats'][0]
    		res_features[b, 1:] = batch_blobs[b]['res50_feats'][2:] # No need to use hull features for GCN spatial subnet
    res_features = torch.FloatTensor(res_features).cuda()

    inputs = [one_hot_encodings, batch_node_bboxes, num_objs, res_features]
    subact_cls_scores, afford_cls_scores = model.forward(inputs) 

    human_loss = criterion(subact_cls_scores, torch.Tensor(label_H).long().cuda())

    valid_labels = []
    for b in range(batchSize):
        for n in range(0, num_objs[b]):
            valid_labels.append(label_O[b][n])
    label_O = torch.Tensor(valid_labels)
    object_loss = criterion(afford_cls_scores, label_O.long().cuda())
    
    loss = human_loss + cfg.TRAIN.OBJECT_LOSS_SCALE*object_loss

    if is_train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    batch_loss += loss.item()
    batch_human_loss += human_loss.item()
    batch_object_loss += object_loss.item() 

    subact_cls_scores = subact_cls_scores.cpu().detach().numpy()
    afford_cls_scores = afford_cls_scores.cpu().detach().numpy()

    h_preds = []
    h_gts = []
    for b in range(batchSize):
        H_pred = np.argmax(subact_cls_scores[b])
        h_preds.append(H_pred)
        h_gts.append(label_H[b])

    o_preds = []
    o_gts = []
    for b in range(label_O.shape[0]):
        O_pred = np.argmax(afford_cls_scores[b])
        o_preds.append(O_pred)
        o_gts.append(label_O[b].item())
        
    return batch_loss, batch_human_loss, batch_object_loss, h_preds, h_gts, o_preds, o_gts

# Preprocess training and testing data
blobs_video_train, blobs_video_test = {}, {}
for i in range(len(blobs_segment_train)):
    vid_id = blobs_segment_train[i]['video_id']
    if vid_id not in list(blobs_video_train.keys()):
        blobs_video_train[vid_id] = []
    blobs_video_train[vid_id].append(blobs_segment_train[i])

for i in range(len(blobs_segment_test)):
    vid_id = blobs_segment_test[i]['video_id']
    if vid_id not in list(blobs_video_test.keys()):
        blobs_video_test[vid_id] = []
    blobs_video_test[vid_id].append(blobs_segment_test[i])

for key in list(blobs_video_train.keys()):
    blobs_video_train[key].sort(key=lambda x : x['seg_frames'][0])
    if cfg.TASK == 'anticipation':
        num_segs = len(blobs_video_train[key])
        for i in range(num_segs-1):
            blobs_video_train[key][i]['label_H'] = blobs_video_train[key][i+1]['label_H']
            blobs_video_train[key][i]['label_O'] = blobs_video_train[key][i+1]['label_O']
        del blobs_video_train[key][num_segs-1]
    
for key in list(blobs_video_test.keys()):
    blobs_video_test[key].sort(key=lambda x : x['seg_frames'][0]) 
    if cfg.TASK == 'anticipation':
        num_segs = len(blobs_video_test[key])
        for i in range(num_segs-1):
            blobs_video_test[key][i]['label_H'] = blobs_video_test[key][i+1]['label_H']
            blobs_video_test[key][i]['label_O'] = blobs_video_test[key][i+1]['label_O']
        del blobs_video_test[key][num_segs-1]

blobs_video_train, blobs_video_test = list(blobs_video_train.values()), list(blobs_video_test.values())
blobs_segment_train = [x for video in blobs_video_train for x in video]
blobs_segment_test = [x for video in blobs_video_test for x in video]

if cfg.TRAIN_LEVEL == 'frame':
    del blobs_video_train, blobs_video_test
elif cfg.TRAIN_LEVEL == 'segment':
    del blobs_segment_train, blobs_segment_test

print('Start training')
F1_score_trace = [0]

for epoch in range(cfg.TRAIN.EPOCHS): 
    log_file = open(os.path.join(cfg.LOG_DIR, 'log_'+experiment_name+'.txt'), "a+")

    # Run on training data
    epoch_start = time.time()
    total_loss, total_human_loss, total_obj_loss = 0.0, 0.0, 0.0
    H_preds, H_gts, O_preds, O_gts = [], [], [], []

    if cfg.TRAIN_LEVEL == 'frame':
        num_batches = math.ceil(len(blobs_segment_train)/cfg.TRAIN.BATCH_SIZE)
        random.shuffle(blobs_segment_train)
    elif cfg.TRAIN_LEVEL == 'segment':
        num_batches = len(blobs_video_train)
        random.shuffle(blobs_video_train)

    model.train()
    start = time.time()
    for i in range(num_batches):
        if (i+1)%100 == 0:
            print('Progress:',str(i+1)+'/'+str(num_batches), 'time:', time.time()-start)
            start = time.time()
        if cfg.TRAIN_LEVEL == 'frame':
            batch_blobs = blobs_segment_train[cfg.TRAIN.BATCH_SIZE*i : cfg.TRAIN.BATCH_SIZE*(i+1)]
        elif cfg.TRAIN_LEVEL == 'segment':
            batch_blobs = blobs_video_train[i]

        batch_loss, batch_human_loss, batch_object_loss, h_preds, h_gts, o_preds, o_gts = forward_step(batch_blobs, True)
        total_loss += batch_loss
        total_human_loss += batch_human_loss
        total_obj_loss += batch_object_loss
        H_preds += h_preds
        O_preds += o_preds
        H_gts += h_gts
        O_gts += o_gts

    total_loss = total_loss / num_batches
    total_human_loss = total_human_loss / num_batches
    total_obj_loss = total_obj_loss / num_batches

    H_gts = list(map(int, H_gts)) 
    O_gts = list(map(int, O_gts)) 

    human_accuracy = 0.0
    for i in range(len(H_gts)):
        if H_gts[i] == H_preds[i]:
            human_accuracy += 1.0
    human_accuracy = 100.0*human_accuracy/len(H_gts)

    object_accuracy = 0.0
    for i in range(len(O_gts)):
        if O_gts[i] == O_preds[i]:
            object_accuracy += 1.0
    object_accuracy = 100.0*object_accuracy/len(O_gts)

    epoch_train_time = time.time()-epoch_start        
    print('Epoch:%02d, loss: %.6f, human_loss: %.6f, object_loss: %.6f, human_acc: %.4f, object_acc: %.4f, lr: %.8f, time: %.3f s/iter' %  
          (epoch, total_loss, total_human_loss, total_obj_loss, human_accuracy, object_accuracy, learning_rate, epoch_train_time))
    log_file.write('Epoch:%02d, loss: %.6f, human_loss: %.6f, object_loss: %.6f, human_acc: %.4f, object_acc: %.4f, lr: %.8f, time: %.3f s/iter\n' %  
          (epoch, total_loss, total_human_loss, total_obj_loss, human_accuracy, object_accuracy, learning_rate, epoch_train_time))

    # Run on testing set
    epoch_start = time.time()
    total_loss, total_human_loss, total_obj_loss = 0.0, 0.0, 0.0

    model.eval()
    H_preds, H_gts, O_preds, O_gts = [], [], [], []

    if cfg.TRAIN_LEVEL == 'frame':
        num_batches = math.ceil(len(blobs_segment_test)/cfg.TRAIN.BATCH_SIZE)
    elif cfg.TRAIN_LEVEL == 'segment':
        num_batches = len(blobs_video_test)

    for i in range(num_batches):
        if cfg.TRAIN_LEVEL == 'frame':
            batch_blobs = blobs_segment_test[cfg.TRAIN.BATCH_SIZE*i : cfg.TRAIN.BATCH_SIZE*(i+1)]
        elif cfg.TRAIN_LEVEL == 'segment':
            batch_blobs = blobs_video_test[i]

        batch_loss, batch_human_loss, batch_object_loss, h_preds, h_gts, o_preds, o_gts = forward_step(batch_blobs, False)
        total_loss += batch_loss
        total_human_loss += batch_human_loss
        total_obj_loss += batch_object_loss
        H_preds += h_preds
        O_preds += o_preds
        H_gts += h_gts
        O_gts += o_gts
   
    H_gts = list(map(int, H_gts)) 
    O_gts = list(map(int, O_gts)) 

    subact_f1_score = sklearn.metrics.f1_score( H_gts, H_preds, labels=range(10), average='macro')*100
    afford_f1_score = sklearn.metrics.f1_score( O_gts, O_preds, labels=range(12), average='macro')*100

    total_loss = total_loss / num_batches
    total_human_loss = total_human_loss / num_batches
    total_obj_loss = total_obj_loss / num_batches

    epoch_test_time = time.time() - epoch_start
    print('Test ' + \
        ', time: %.3f'%(epoch_test_time) + \
        ', subact_fmacro: %.5f'%(subact_f1_score) + \
        ', afford_fmacro: %.5f'%(afford_f1_score)) 
    log_file.write('Test ' + \
        ', time: %.3f'%(epoch_test_time) + \
        ', subact_F1: %.5f'%(subact_f1_score) + \
        ', afford_F1: %.5f\n'%(afford_f1_score))

    # learning rate decay, saving checkpoint

    if (epoch+1)%cfg.TRAIN.STEPSIZE == 0:
        if learning_rate > 1e-7:
            learning_rate *= cfg.TRAIN.WEIGHT_DECAY
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            print('decrease lr to', learning_rate)

    if subact_f1_score + afford_f1_score > F1_score_trace[-1] and epoch >= 5 :
        F1_score_trace.append(subact_f1_score + afford_f1_score)
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

