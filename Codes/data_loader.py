# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:04:24 2023

@author: Omar Al-maqtari
"""

import os
import cv2
import random
from PIL import Image
from perlin import rand_perlin_2d

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F


class ImageFolder(Dataset):
    def __init__(self, cfg, mode, aug_prob):
        """Initializes image paths and preprocessing module."""
        self.root = cfg.dataset_path
		
		# GT : Ground Truth
        self.image_paths = sorted(list(os.listdir(self.root+mode+'/')))
        self.TSGT_paths = sorted(list(os.listdir(self.root+mode+'_lab/')))
        
        self.image_height = cfg.image_height
        self.image_width = cfg.image_width
        self.training_mode = cfg.training_mode
        self.mode = mode
        self.aug_prob = aug_prob
        self.use_mask = cfg.use_mask
        self.bg_threshold = cfg.bg_threshold
        self.bg_reverse = cfg.bg_reverse
        self.switch = True
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))
    
    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        TSGT_path = self.TSGT_paths[index]
        
        image = Image.open(self.root+self.mode+'/'+image_path)
        TSGT = Image.open(self.root+self.mode+'_lab/'+TSGT_path)
        
        sharpness_factor = random.choice([2,3,4])
        Transform = []
        Transform.append(T.Resize((self.image_height,self.image_width)))
        Transform.append(T.RandomAdjustSharpness(sharpness_factor=sharpness_factor,p=0.25))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        image = Transform(image)
        TSGT = Transform(TSGT)[:1,:]
        if TSGT.sum() > 1.:
            TCGT = torch.tensor(float('1'), dtype=torch.long)
        else:
            TCGT = torch.tensor(float('0'), dtype=torch.long)
        
        if image.shape[0] < 3:
            image = image.repeat(3,1,1)
            
        if self.mode == 'train':
            if random.random() < 0.5:
                Transform = T.RandomRotation((90,90),expand=True)
                image = Transform(image)
                TSGT = Transform(TSGT)
                
            if random.random() < 0.5:
                image = F.hflip(image)
                TSGT = F.hflip(TSGT)
                
            if random.random() < 0.5:
                image = F.vflip(image)
                TSGT = F.vflip(TSGT)
                
            if self.training_mode == 'FSS i' and random.random() < self.aug_prob:
                image, TSGT = self.generate_anomaly(image)
                
                TCGT = torch.tensor(float('1'), dtype=torch.long)
                
            if self.training_mode == 'FSS l' and self.switch:
                image, TSGT = self.generate_anomaly(image)
                
                TCGT = torch.tensor(float('1'), dtype=torch.long)
                
                self.switch = False
            else: self.switch = True
            
            Transform = []
            Transform = T.Resize((self.image_height,self.image_width))
            image = Transform(image)
            TSGT = Transform(TSGT)
        
        if TSGT.shape[0] > 1:
            Transform = T.Grayscale(num_output_channels=1)
            TSGT = Transform(TSGT)
            
        TSGT[TSGT<.5] = 0.
        TSGT[TSGT>=.5] = 1.
        
        return image, [TCGT, TSGT], image_path

    def generate_anomaly(self, image):
        
        if self.use_mask:
            target_foreground = self.target_foreground_mask(image)
        else:
            target_foreground = torch.ones(image.shape[1:])
        
        perlin_scalex = 2**(torch.randint(0, 4, (1,)).numpy()[0])
        perlin_scaley = 2**(torch.randint(0, 4, (1,)).numpy()[0])
        perlin_noise = rand_perlin_2d((self.image_height, self.image_width), (perlin_scalex, perlin_scaley)).unsqueeze(0)
        threshold_factor = random.choice([.45,.5,.55])
        perlin_noise[perlin_noise<threshold_factor] = 0.
        perlin_noise[perlin_noise>=threshold_factor] = 1.
        
        GT = perlin_noise * target_foreground
        GT_expanded = GT.repeat(3,1,1)
        
        anomaly_factor = random.uniform(.8, 1.)
        texture_img = T.RandomRotation((-90,90),expand=False)(image)
        anomaly_source_img = (anomaly_factor * (GT_expanded * texture_img)) + ((1 - anomaly_factor) * GT_expanded)
        Elastic_factor = random.choice([8.,9.,10.])
        anomaly_source_img = T.ElasticTransform(alpha=90.0, sigma=Elastic_factor)(anomaly_source_img)
        
        anomaly_source_img = ((-GT_expanded + 1) * image) + anomaly_source_img
        
        return anomaly_source_img, GT

    def target_foreground_mask(self, image):
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(image.permute(1,2,0).numpy(), cv2.COLOR_RGB2GRAY)*255
        
        # generate binary mask of gray scale image
        _, target_foreground_mask = cv2.threshold(img_gray, self.bg_threshold, 255, cv2.THRESH_BINARY)
        
        # invert mask for foreground mask
        if self.bg_reverse:
            target_foreground_mask = 255.0 - target_foreground_mask
        else:
            target_foreground_mask = target_foreground_mask
        
        return torch.from_numpy(target_foreground_mask).unsqueeze(0)/255.0


def get_loader(cfg, mode, aug_prob):
    """Builds and returns Dataloader."""
    
    dataset = ImageFolder(cfg, mode, aug_prob)
    
    data_loader = DataLoader(dataset=dataset,
                             batch_size=cfg.batch_size,
                             shuffle=True if mode=='train' else False,
                             num_workers=cfg.num_workers,
                             pin_memory=True,
                             drop_last=True)
    
    return data_loader
