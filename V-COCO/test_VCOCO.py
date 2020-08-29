import sys
import torch
import torch.nn as nn
import time
import numpy as np
import scipy.io as sio
import pickle as pkl
import cv2
import math
import os
import h5py
import json

from config import cfg
sys.path.append(os.path.join(cfg.PARENT_DIR, 'models'))
from skimage.transform import resize
from model_VCOCO_LIGHTEN import LIGHTEN_image
from apply_prior import apply_prior
from vsrl_eval import VCOCOeval

####################################
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print('added {} to pythonpath'.format(path))

coco_path = os.path.join(cfg.COCO_PATH, 'coco', 'PythonAPI')
add_path(coco_path)

from pycocotools.coco import COCO 
####################################
   
def func_object(prob):
    return (np.exp(8*prob)-1)/(np.exp(8)-1)

def get_blob(image_id):
    im_file  = os.path.join(cfg.VCOCO_IMAGE_DIR + str(image_id).zfill(12) + '.jpg')
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)/256.0
    im_shape = im_orig.shape
    return im_orig, im_shape

save_path = os.path.join(cfg.PARENT_DIR, 'output', 'vcoco_test_scores.pkl')

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

model = LIGHTEN_image(spatial_subnet=cfg.SPATIAL_SUBNET, drop=0.0)
res50 = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
res50 = torch.nn.Sequential(*(list(res50.children())[:-1]))
res50.cuda()
res50.eval()
sigmoid = nn.Sigmoid()

model.float().cuda()     
model.eval()

# set to true if test scores are already computed
do_eval_only = False

if do_eval_only:
    vcocoeval = VCOCOeval(os.path.join(cfg.VCOCO_EVAL_DIR, 'vcoco/vcoco_test.json'), os.path.join(cfg.VCOCO_EVAL_DIR, 'instances_vcoco_all_2014.json'), os.path.join(cfg.VCOCO_EVAL_DIR, 'splits/vcoco_test.ids') ) 
    vcocoeval._do_eval(save_path, ovr_thresh=0.5) 
    print(save_path)
    exit(1)

experiment_name = cfg.SPATIAL_SUBNET
if cfg.TRAIN_VAL :
    experiment_name += '_trainVal'
checkpoint_file = os.path.join(cfg.CHECKPOINT_DIR, 'checkpoint_'+experiment_name+'.pth')

if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded model from checkpoint')
else:
    print('Checkpoint file does not exist')
    exit(1)

meta = h5py.File(os.path.join(cfg.VCOCO_DIR, 'devkit/vcoco_meta.h5'),'r')

Test_RCNN_Keypoints  = pkl.load( open(os.path.join(cfg.DATA_DIR, 'Test_Faster_RCNN_R-50-PFN_2x_VCOCO_Keypoints.pkl'), "rb" ), encoding='latin1' )

prior_mask = pkl.load( open( os.path.join(cfg.DATA_DIR, 'prior_mask.pkl'), 'rb'), encoding='latin1' )
Action_dic =json.load( open( os.path.join(cfg.DATA_DIR, 'action_index.json')))
Action_dic_inv = {y:x for x,y in Action_dic.items()}
hoi_list = {0: 'surf_instr', 1: 'ski_instr', 2: 'cut_instr', 3: 'walk', 4: 'cut_obj', 5: 'ride_instr', 6: 'talk_on_phone_instr', \
7: 'kick_obj', 8: 'work_on_computer_instr', 9: 'eat_obj', 10: 'sit_instr', 11: 'jump_instr', 12: 'lay_instr', 13: 'drink_instr', \
14: 'carry_obj', 15: 'throw_obj', 16: 'eat_instr', 17: 'smile', 18: 'look_obj', 19: 'hit_instr', 20: 'hit_obj', 21: 'snowboard_instr', \
22: 'run', 23: 'point_instr', 24: 'read_obj', 25: 'hold_obj', 26: 'skateboard_instr', 27: 'stand', 28: 'catch_obj'} # the same order 29 HOI GTs in iCAN

hoilist = [] # convert 26 to 29
for i, hoi in hoi_list.items():
    idx = [j for j in range(26) if str(meta['meta/pre/idx2name/' + str(j)][...])[2:-1] in hoi][0]
    hoilist.append(idx)

#####################################################

def crop_pool_norm(batch_images, batch_boxes):
    batch_segments = np.zeros([len(batch_images), 224, 224, 3])*1.0

    st = time.time()
    for i in range(len(batch_images)):
        box = batch_boxes[i]
        x1, y1, x2, y2 = math.floor(box[0]), math.floor(box[1]), math.floor(box[2]), math.floor(box[3])
        batch_cropped = batch_images[i][y1:y2, x1:x2, :]

        batch_segments[i] = resize(batch_cropped, (224, 224, 3))

    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    batch_segments = (batch_segments - img_mean)/img_std
    return batch_segments

# timers
cnt=0
result = []

for imid in Test_RCNN_Keypoints.keys(): 
    st_time = time.time()
    print('Image_id', imid)     

    all_detections = Test_RCNN_Keypoints[imid]
    human_detections, object_detections = [], []
    for detect in all_detections:
        if detect[1] == 'Human' and (np.max(detect[5]) > cfg.TEST.HUMAN_THRES):
            human_detections.append(detect)
        elif detect[1] == 'Object' and (np.max(detect[5]) > cfg.TEST.OBJECT_THRES):
            object_detections.append(detect)

    print(len(human_detections), len(object_detections))

    im_orig, im_shape = get_blob(imid)


    for Human_out in human_detections: # detected humans
        scoreh = func_object(Human_out[5]) 
        # print('scoreh', scoreh)
        box_H = np.array([Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,4)
        ###########################
        
        # save image information
        dic = {}
        dic['image_id']   = str(imid).zfill(6)
        dic['person_box'] = Human_out[2]

        # Predict action using human and object appearance 
        Score_obj     = np.empty((0, 4 + 29), dtype=np.float32) 

        batch_images = [im_orig]
        batch_human_boxes = torch.FloatTensor(box_H).cuda()
        batch_human_segments = crop_pool_norm(batch_images, box_H)
        batch_human_segments = torch.FloatTensor(batch_human_segments).cuda().permute(0, 3, 1, 2).contiguous()

        batch_res_human_feats = res50(batch_human_segments)
        # ctr = 0

        for Object in object_detections: # detected objects

            # print(Object)
            scoreo = func_object(Object[5])
            # print('scoreo', scoreo)
            
            box_O = np.array([Object[2][0],Object[2][1],Object[2][2],Object[2][3]]).reshape(1,4)
            
            # print(imid, box_H, im_orig.shape)
            batch_object_segments = crop_pool_norm(batch_images, box_O)

            batch_object_segments = torch.FloatTensor(batch_object_segments).cuda().permute(0, 3, 1, 2).contiguous()
            batch_object_boxes = torch.FloatTensor(box_O).cuda()

            batch_res_object_feats = res50(batch_object_segments)

            action_raw_scores = \
                model.forward(batch_human_boxes, batch_object_boxes, \
                    batch_res_human_feats, batch_res_object_feats)

            action_cls_scores = sigmoid(action_raw_scores)
            
            probs = action_cls_scores.cpu().detach().numpy()
            probs_H = action_cls_scores.cpu().detach().numpy()
            
            # convert to len=29
            prediction_HO = [probs[0,idx] for idx in hoilist] 
            prediction_H  = [probs_H[0,idx] for idx in hoilist]                        
            
            # filter 29 verb predictions based on object category
            cfg.TEST.PRIOR_FLAG = 3
            if cfg.TEST.PRIOR_FLAG == 1:
                prediction_HO  = apply_prior(Object[4], prediction_HO)
                prediction_HO = 1.0*np.array(prediction_HO).reshape(1, 29)
            if cfg.TEST.PRIOR_FLAG == 2:
                prediction_HO  = prediction_HO * prior_mask[:,Object[4]].reshape(1,29) 
            if cfg.TEST.PRIOR_FLAG == 3:
                prediction_HO  = apply_prior(Object[4], prediction_HO)
                prediction_HO  = prediction_HO * prior_mask[:,Object[4]].reshape(1,29)
            prediction_H = np.array(prediction_H).reshape(1,29)
            
            
            This_Score_obj = np.concatenate((Object[2].reshape(1,4), prediction_HO * np.max(scoreo)), axis=1)   
            Score_obj      = np.concatenate((Score_obj, This_Score_obj), axis=0)
                
        # print('Objects over')
        if Score_obj.shape[0] == 0:
            continue
        
        # Find out the object box associated with highest action score
        max_idx = np.argmax(Score_obj,0)[4:] 

        # agent mAP
        for i in range(29):
            #'''
            # walk, smile, run, stand
            if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                agent_name      = Action_dic_inv[i] + '_agent'
                dic[agent_name] = np.max(scoreh) * prediction_H[0, i]
                continue

            # cut
            if i == 2:
                agent_name = 'cut_agent'
                dic[agent_name] = np.max(scoreh) * max(Score_obj[max_idx[2]][4 + 2], Score_obj[max_idx[4]][4 + 4])
                continue 
            if i == 4:
                continue   

            # eat
            if i == 9:
                agent_name = 'eat_agent'
                dic[agent_name] = np.max(scoreh) * max(Score_obj[max_idx[9]][4 + 9], Score_obj[max_idx[16]][4 + 16])
                continue  
            if i == 16:
                continue

            # hit
            if i == 19:
                agent_name = 'hit_agent'
                dic[agent_name] = np.max(scoreh) * max(Score_obj[max_idx[19]][4 + 19], Score_obj[max_idx[20]][4 + 20])
                continue  
            if i == 20:
                continue  

            # These 2 classes need to save manually because there is '_' in action name
            if i == 6:
                agent_name = 'talk_on_phone_agent'  
                dic[agent_name] = np.max(scoreh) * Score_obj[max_idx[i]][4 + i]
                continue

            if i == 8:
                agent_name = 'work_on_computer_agent'  
                dic[agent_name] = np.max(scoreh) * Score_obj[max_idx[i]][4 + i]
                continue 

            # all the rest
            agent_name =  Action_dic_inv[i].split("_")[0] + '_agent'  
            dic[agent_name] = np.max(scoreh) * Score_obj[max_idx[i]][4 + i]
            #'''

        # role mAP
        for i in range(29):
            # walk, smile, run, stand. Won't contribute to role mAP
            if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), np.max(scoreh) * prediction_H[0, i]) 
                continue

            # Impossible to perform this action based on prior filters
            if np.max(scoreh) * Score_obj[max_idx[i]][4 + i] == 0:
                dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), np.max(scoreh) * Score_obj[max_idx[i]][4 + i])

            # Action with >0 score
            else:
                dic[Action_dic_inv[i]] = np.append(Score_obj[max_idx[i]][:4], np.max(scoreh) * Score_obj[max_idx[i]][4 + i])

        result.append(dic)
                
    print('im_detect: {:d}/{:d} {:.3f}s'.format(cnt + 1, 4946, time.time()-st_time))
    cnt+=1
   
with open(save_path, 'wb+') as f:
    pkl.dump(result, f, pkl.HIGHEST_PROTOCOL) 

# Evaluation 
vcocoeval = VCOCOeval(os.path.join(cfg.VCOCO_EVAL_DIR, 'vcoco/vcoco_test.json'), os.path.join(cfg.VCOCO_EVAL_DIR, 'instances_vcoco_all_2014.json'), os.path.join(cfg.VCOCO_EVAL_DIR, 'splits/vcoco_test.ids') ) 
vcocoeval._do_eval(save_path, ovr_thresh=0.5) 
print(save_path)
