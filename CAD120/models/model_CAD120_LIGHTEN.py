import os
import torch
import torch.nn as nn
import sys
import numpy as np

from config import cfg
from GCN_spatial_subnet import GCN_spatial_subnet

sys.path.append(os.path.join(cfg.PARENT_DIR, 'models'))

class LIGHTEN_frame_level(nn.Module):
    def __init__(self, spatial_subnet='', dropout=0.0):
        super(LIGHTEN_frame_level, self).__init__()
        self.subact_classes = 10
        self.afford_classes = 12
        self.res_feat_dim = 2048
        self.num_frames = 20
        self.num_nodes = 6
        self.dropout = dropout
        self.spatial_subnet_type = spatial_subnet

        if self.spatial_subnet_type == 'GCN':
            self.preprocess_dim = 1000
            self.hidden_dim = 512
            self.out_dim = 512

        self.preprocess_human = nn.Linear(self.res_feat_dim, self.preprocess_dim)
        self.preprocess_object = nn.Linear(self.res_feat_dim, self.preprocess_dim)

        if self.spatial_subnet_type == 'GCN':
            self.in_dim = 8 + 2*self.preprocess_dim + 10
            # Initialization to Adjacency matrix
            self.A = np.zeros((6, 6))
            self.A[0, :] = 1
            self.A[:, 0] = 1
            for i in range(6):
                self.A[i, i] = 1
            self.spatial_subnet = GCN_spatial_subnet(self.in_dim, self.hidden_dim, self.out_dim, self.A)

        # RNN blocks for frame-level temporal subnet
        self.subact_frame_RNN = nn.RNN(input_size=self.out_dim, hidden_size=self.out_dim//2, num_layers=2, batch_first=True, bidirectional=True)
        self.afford_frame_RNN = nn.RNN(input_size=2*self.out_dim, hidden_size=self.out_dim, num_layers=2, batch_first=True, bidirectional=True)

        self.classifier_human = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim), 
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.subact_classes)
        )
        self.classifier_object = nn.Sequential(
            nn.Linear(2*self.out_dim, self.out_dim), 
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.afford_classes)
        )
    
    def forward(self, inputs, out_type='scores'):
        # Graph node feature initialization
        one_hot_encodings, node_bboxes, num_objs, res_features = inputs[0], inputs[1], inputs[2], inputs[3]
        batchSize = node_bboxes.shape[0]

        if self.spatial_subnet_type == 'GCN':
            node_feats = torch.zeros((batchSize, 6, 20, self.in_dim)).float().cuda()
            
            node_feats[:, 0, :, :4] = node_bboxes[:, 0, :, :]

            human_precomp_feats = torch.flatten(res_features[:, 0, :, :, :, :].reshape(batchSize*self.num_frames, self.res_feat_dim, 1, 1), 1)
            node_feats[:, 0, :, 4:4+self.preprocess_dim] = self.preprocess_human(human_precomp_feats).reshape(batchSize, self.num_frames, self.preprocess_dim)
           
            obj_precomp_feats = torch.flatten(res_features[:, 1:, :, :, :, :].reshape(batchSize*5*self.num_frames, self.res_feat_dim, 1, 1), 1)
            node_feats[:, 1:, :, 4+self.preprocess_dim:4+2*self.preprocess_dim] = self.preprocess_object(obj_precomp_feats).reshape(batchSize, 5, self.num_frames, self.preprocess_dim)

            node_feats[:, 1:, :, 4+2*self.preprocess_dim:8+2*self.preprocess_dim] = node_bboxes[:, 1:, :, :]
            node_feats[:, 1:, :, 8+2*self.preprocess_dim:] = one_hot_encodings[:, :, :, :]

            node_feats = node_feats.permute(0, 2, 1, 3)
            node_feats = node_feats[:, :, :, :].reshape(batchSize*self.num_frames, 1, 6, self.in_dim)

        # Spatial Subnet
        # Output of spatial subnet:  (batchSize, self.num_frames, 6, self.out_dim)
        if self.spatial_subnet_type == 'GCN':
            spatial_graph = self.spatial_subnet(node_feats.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(batchSize, self.num_frames, 6, self.out_dim)

        # Append human feats to object nodes
        human_node_feats = spatial_graph[:, :, 0, :]
                
        total_obj = sum(num_objs)
        count = 0
        obj_node_feats = []
        for b in range(batchSize):
            obj_feats = spatial_graph[b, :, 1: 1+num_objs[b], :]

            concat_feats = torch.zeros((self.num_frames, num_objs[b], 2*self.out_dim)).float().cuda()
            for o in range(num_objs[b]):
                concat_feats[:, o, :] = torch.cat((human_node_feats[b, :, :], obj_feats[:, o, :]), 1)

            obj_node_feats.append(concat_feats)
            
        obj_node_feats = torch.cat(obj_node_feats, dim=1)
        obj_node_feats = obj_node_feats.permute(1, 0, 2) #total_obj x num_frames x 128

        ## Frame-level Temporal subnet
        human_rnn_feats = self.subact_frame_RNN(human_node_feats, None)[0]
        obj_rnn_feats = self.afford_frame_RNN(obj_node_feats, None)[0]

        subact_cls_scores = torch.sum(self.classifier_human(human_rnn_feats), dim=1)
        afford_cls_scores = torch.sum(self.classifier_object(obj_rnn_feats), dim=1)

        if out_type == 'scores': 
            return subact_cls_scores, afford_cls_scores
        elif out_type == 'seg_feats':
            return human_rnn_feats, obj_rnn_feats


class LIGHTEN_segment_level(nn.Module):
    def __init__(self, spatial_subnet='', dropout=0.0):
        super(LIGHTEN_segment_level, self).__init__()

        self.frame_level_model = LIGHTEN_frame_level(spatial_subnet=spatial_subnet, dropout=dropout)
        self.dropout = dropout

        # Initialize parameters using checkpoint from training frame-level subnet
        # frame_checkpoint = torch.load(os.path.join(cfg.CHECKPOINT_DIR, 'checkpoint_'+spatial_subnet+'_frame_'+cfg.TASK+'.pth'))
        frame_checkpoint = torch.load(os.path.join(cfg.CHECKPOINT_DIR, 'checkpoint_'+spatial_subnet+'_frame_detection.pth'))
        self.frame_level_model.load_state_dict(frame_checkpoint['state_dict'])

        self.out_dim = self.frame_level_model.out_dim
        self.subact_classes = self.frame_level_model.subact_classes
        self.afford_classes = self.frame_level_model.afford_classes
        self.num_frames = self.frame_level_model.num_frames

        # MLP to compute attention weights for aggregation
        self.subact_attention_model = nn.Sequential(
            nn.Linear(self.num_frames*self.out_dim, self.num_frames*16),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.num_frames*16, self.num_frames),
        )
        self.afford_attention_model = nn.Sequential(
            nn.Linear(self.num_frames*2*self.out_dim, self.num_frames*32),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.num_frames*32, self.num_frames),
        )

        # Segment level subnet
        self.subact_segment_level_subnet = nn.Sequential(
            nn.RNN(input_size=self.out_dim, hidden_size=self.out_dim, num_layers=3, batch_first=True, dropout=self.dropout, bidirectional=False),
        )
        self.afford_segment_level_subnet = nn.Sequential(
            nn.RNN(input_size=2*self.out_dim, hidden_size=2*self.out_dim, num_layers=3, batch_first=True, dropout=self.dropout, bidirectional=False),
        )
        self.seg_human_classifier = nn.Linear(self.out_dim, self.subact_classes)
        self.seg_object_classifier = nn.Linear(2*self.out_dim, self.afford_classes)

    def forward(self, inputs):
        batchSize = inputs[1].shape[0]
        num_objs = inputs[2][0]

        subact_frame_embeddings, afford_frame_embeddings = self.frame_level_model.forward(inputs, out_type='seg_feats')
        subact_frame_embeddings = subact_frame_embeddings.reshape(batchSize, self.num_frames, self.out_dim)
        afford_frame_embeddings = afford_frame_embeddings.reshape(batchSize*num_objs, self.num_frames, 2*self.out_dim)

        subact_attention_map = self.subact_attention_model(subact_frame_embeddings.reshape(batchSize, self.num_frames*self.out_dim))
        afford_attention_map = self.afford_attention_model(afford_frame_embeddings.reshape(batchSize*num_objs, self.num_frames*2*self.out_dim))
        subact_segment_embeddings = torch.sum(subact_attention_map.unsqueeze(2).expand(batchSize, self.num_frames, self.out_dim)*subact_frame_embeddings, dim=1)
        afford_segment_embeddings = torch.sum(afford_attention_map.unsqueeze(2).expand(batchSize*num_objs, self.num_frames, 2*self.out_dim)*afford_frame_embeddings, dim=1)

        subact_segment_feats = self.subact_segment_level_subnet(subact_segment_embeddings.reshape(1, batchSize, self.out_dim))[0]
        subact_cls_scores = self.seg_human_classifier(subact_segment_feats.reshape(batchSize, self.out_dim))

        afford_segment_feats = self.afford_segment_level_subnet(afford_segment_embeddings.reshape(batchSize, num_objs, 2*self.out_dim)\
            .permute(1, 0, 2))[0].permute(1, 0, 2)

        afford_cls_scores = self.seg_object_classifier(afford_segment_feats.reshape(batchSize*num_objs, 2*self.out_dim))

        return subact_cls_scores, afford_cls_scores