# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 16:25:25 2023

Modified by Omar Al-maqtari
"""

from math import pi

import torch
from torch import nn

class FFM_Net(nn.Module):
    def __init__(self, ch_in, ch_out, mapping_size, num_layers, scale=10):
        super(FFM_Net, self).__init__()
        self.mapping_size = (mapping_size//2)
        self.B = torch.randn(ch_in, (mapping_size//2)) * scale
        
        self.FFM_Net = []
        for i in range(num_layers-1):
            self.FFM_Net.append(nn.Conv2d(mapping_size,mapping_size,kernel_size=1,padding=0, groups=mapping_size//4))
            self.FFM_Net.append(nn.ReLU())
            self.FFM_Net.append(nn.BatchNorm2d(mapping_size))
            
        self.FFM_Net.append(nn.Dropout(0.1))
        self.FFM_Net.append(nn.Conv2d(mapping_size,ch_out,kernel_size=1,padding=0))
        self.FFM_Net.append(nn.Sigmoid())
        self.FFM_Net = nn.Sequential(*self.FFM_Net)
        
    def forward(self, x):
        batches, channels, width, height = x.shape
        
        x = x.permute(0, 2, 3, 1).reshape(batches, width * height, channels)
        x = (2*pi*x) @ self.B.to(x.device)
        x = x.view(batches, width, height, self.mapping_size).permute(0, 3, 1, 2)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=1).to(x.device)
        
        return self.FFM_Net(x)
    