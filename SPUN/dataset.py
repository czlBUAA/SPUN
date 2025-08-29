import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import glob
import pandas as pd
from PIL import Image
import numpy as np
import utils
from config import cfg

    
class SPEdataset(Dataset):
    def __init__(self, image_root, subset, transform=None):
        set_filename = image_root + '/' + subset + '_images.csv'
        rgb_list_df = pd.read_csv(set_filename, names=['filename'], header=None)
        rgb_list = list(rgb_list_df['filename'])
        self.image_root = rgb_list
        self.root = image_root
        self.label = pd.read_csv(image_root + '/' + subset + '_poses_gt.csv')
        self.transforms = transform
        self.len = len(rgb_list)

    def __getitem__(self, index):
        image = Image.open(self.root + '/' + self.image_root[index]).convert('RGB')
        
        # image = np.array(image)
        # image, window, scale, padding, crop = utils.resize_image(
        # image,
        # min_dim=512,
        # min_scale=0,
        # max_dim=640,
        # mode='pad64')
        # image = Image.fromarray(image.astype(np.uint8))
        if self.label['q4'][index] < 0:
            quater = np.asarray([-self.label['q1'][index], -self.label['q2'][index], -self.label['q3'][index], -self.label['q4'][index]])
        else:
            quater = np.asarray([self.label['q1'][index], self.label['q2'][index], self.label['q3'][index], self.label['q4'][index], ])
        pos = np.asarray([self.label['x'][index], self.label['y'][index], self.label['z'][index]])
        if cfg.rot_image:
            dice = np.random.rand(1)
            # Camera orientation perturbation half the time
            fov_x = 90.0 * np.pi / 180
            fov_y = 73.7 * np.pi / 180
            width = 1280  # number of horizontal[pixels]
            height = 960  # number of vertical[pixels]
            # Focal lengths
            fx = width / (2 * np.tan(fov_x / 2))
            fy = - height / (2 * np.tan(fov_y / 2))
            K = np.matrix([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])
            if dice > 0.5:
                image, pos, quater = utils.rotate_cam(image, pos, quater, K, 20)
            elif dice <= 0.5:
                image, pos, quater = utils.rotate_image(image, pos, quater, K)
        image = self.transforms(image)
        quater = np.asarray([quater[3], quater[0],quater[1],quater[2]])
        sample = {'image': image,
                  'quater': torch.from_numpy(quater),
                 'pos': torch.from_numpy(pos)}
        return sample
    def __len__(self):
        return self.len
    