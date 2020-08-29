import os
import torch
import torch.nn as nn
import numpy as np

from config import cfg

from GCN_spatial_subnet import GCN_spatial_subnet

class Edge_Classifier(nn.Module):
    def __init__(self, out_dim, drop,
                    num_classes):
        super(Edge_Classifier, self).__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.drop = drop
        
        self.classifier = nn.Sequential(
            nn.Linear(2*self.out_dim, 2*self.out_dim),
            nn.Dropout(self.drop),
            nn.ReLU(), 
            nn.Linear(2*self.out_dim, self.out_dim),
            nn.Dropout(self.drop),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim),
            nn.Dropout(self.drop),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.num_classes),
        )

    def forward(self, input):
        output = self.classifier(input)
        return output

class LIGHTEN_image(nn.Module):
    def __init__(self, spatial_subnet='', drop=0.0):
        super(LIGHTEN_image, self).__init__()
        self.action_classes = 26
        self.max_num_nodes = 2
        self.spatial_subnet_type = spatial_subnet
        self.drop = drop

        if self.spatial_subnet_type == 'GCN':
            self.preprocess_dim = 1000
            self.in_dim = 2008
            self.hid_dim = 512
            self.out_dim = 512

        self.human_preprocess = nn.Linear(2048, self.preprocess_dim)
        self.object_preprocess = nn.Linear(2048, self.preprocess_dim)
        
        if self.spatial_subnet_type == 'GCN':
            self.A = np.zeros((self.max_num_nodes, self.max_num_nodes))
            self.A[0, :] = 1
            self.A[:, 0] = 1
            for i in range(self.max_num_nodes):
                self.A[i, i] = 1
            self.spatial_subnet = GCN_spatial_subnet(self.in_dim, self.hid_dim, self.out_dim, self.A)
            self.action_classifier = Edge_Classifier(self.out_dim, self.drop, self.action_classes)


    def forward(self, batch_human_boxes, batch_object_boxes, batch_res_human_feats, batch_res_object_feats):
        batchSize = len(batch_human_boxes)
        
        if self.spatial_subnet_type == 'GCN':
            node_feats = torch.zeros(batchSize, self.max_num_nodes, self.in_dim).float().cuda()

            node_feats[:, 0, 4:4+self.preprocess_dim] = self.human_preprocess(torch.flatten(batch_res_human_feats, 1))
            node_feats[:, 1, 4+self.preprocess_dim:-4] = self.object_preprocess(torch.flatten(batch_res_object_feats, 1))

            batch_human_boxes[:, 0], batch_human_boxes[:, 2] = batch_human_boxes[:, 0]/640, batch_human_boxes[:, 2]/640
            batch_human_boxes[:, 1], batch_human_boxes[:, 3] = batch_human_boxes[:, 1]/480, batch_human_boxes[:, 3]/480
            node_feats[:, 0, :4] = batch_human_boxes

            batch_object_boxes[:, 0], batch_object_boxes[:, 2] = batch_object_boxes[:, 0]/640, batch_object_boxes[:, 2]/640
            batch_object_boxes[:, 1], batch_object_boxes[:, 3] = batch_object_boxes[:, 1]/480, batch_object_boxes[:, 3]/480
            node_feats[:, 1, -4:] = batch_object_boxes

            input_graph = node_feats[:, :, :].reshape(batchSize, 1, self.max_num_nodes, self.in_dim)
            gcn_output = self.spatial_subnet(input_graph.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(batchSize, self.max_num_nodes, self.out_dim)
            edge_feats = gcn_output.reshape(batchSize, 2*self.out_dim)

        action_cls_scores = self.action_classifier(edge_feats)
        return action_cls_scores