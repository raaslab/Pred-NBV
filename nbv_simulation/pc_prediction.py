import sys
sys.path.append('/home/user/Code/PoinTr/')

from tools import test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
import torch.nn as nn
import json
from tools import builder
import cv2
import numpy as np

class PC_Predictor():
    def __init__(self, model_path, config_path):
        self.config = cfg_from_yaml_file(config_path)
        self.base_model = builder.model_builder(self.config['model'])
        builder.load_model(self.base_model, model_path, logger = None)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model.to(self.device)
        
        self.base_model.eval()  # set model to eval mode
        self.target = './vis'
        self.useful_cate = [
            "02691156", #plane
            "04379243", #table
            "03790512", #motorbike
            "03948459", #pistol
            "03642806", #laptop
            "03467517", #guitar
            "03261776", #earphone
            "03001627", #chair
            "02958343", #car
            "04090263", #rifle
            "03759954", # microphone
            ]
        
    '''
    def normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        
        return pc, centroid, m
    '''
    
    def normalize(self, pc):
        centroid = torch.mean(pc, dim=1)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc**2, dim=2)))
        pc = pc / m
        return pc, centroid, m
    
    def denormalize(self, pc, centroid, m):
        return (pc * m) + centroid
    
    def predict(self, point_cloud):
        ## If not tensor, convert to tensor
        if isinstance(point_cloud, np.ndarray):
            points = torch.FloatTensor(point_cloud)
        
        ## If not 3-dim, adda dim (batch size)
        if len(points.shape) == 2:
            points = torch.unsqueeze(points, 0)
        
        ## Send to GPU
        points = points.to(self.device)
        
        ## Nromalization
        points, centroid, max_pt = self.normalize(points)
        
        ## Downsampling
        if points.shape[1] > 2048:
            points_down = misc.fps(points, 2048)
        else:
            points_down = points
        
        ## Run model
        ret = self.base_model(points_down)
        coarse_points = ret[0]
        dense_points = ret[-1]
        
        ## Denormalize
        points = self.denormalize(dense_points, centroid, max_pt)
        
        ## Remove batch-dim and convert to numpy
        points_np = points.squeeze().detach().cpu().numpy()
        
        return points_np
    

"""
if __name__ == "__main__":
    pointr_model = PC_Predictor(model_path='/home/dhami/Code/PoinTr/pretrained/90_0.25.pth')
    point_cloud = np.load('/home/dhami/Code/PoinTr/data/ShapeNet55-34/shapenet_pc/02828884-3d2ee152db78b312e5a8eba5f6050bab.npy')
    
"""    