# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:04:24 2023

@author: Omar Al-maqtari
"""

import random

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from einops import rearrange

from mamba import Mamba
from FeatureExtraction import FeatureExtraction
from FFM import FFM_Net

      
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
        
    def forward(self, x):
        return x * self.relu6(x + 3) / 6
    
    
# Squeeze and Excitation Attention
class SEM(nn.Module):
    def __init__(self, ch_out, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out//reduction, kernel_size=1,bias=False),
            nn.ReLU(True),
            nn.Conv2d(ch_out//reduction, ch_out, kernel_size=1,bias=False),
            nn.Sigmoid()
            )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        
        return x * y.expand_as(x)


# Convolution layer
class iConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, groups, bias=False):
        super().__init__()
        self.iconv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel, padding="same", groups=groups, bias=bias),
            nn.GroupNorm(2, ch_out)
            )
            
    def forward(self, x):
        return self.iconv(x)

             
# Feature Decoding Module
class FDM(nn.Module):
    def __init__(self, ch_in, ch_out, reduction):
        super(FDM, self).__init__()
        
        # reduce and concentrate features
        self.squeeze = iConv(ch_in, ch_out, 1, ch_out//2)
        
        # 1x1 conv branch
        self.c1 = iConv(ch_out, ch_out, 1, ch_out//4)
        
        # 3x3 conv branch
        self.c2 = iConv(ch_out, int(ch_out*3), 3, ch_out//4)
        
        self.out = nn.Sequential(
            nn.MaxPool2d(3,1,1),
            nn.Conv2d(ch_in, ch_in, kernel_size=1,groups=ch_out//2,bias=True),
            nn.GroupNorm(2, ch_in)
            )
        
        self.sem1 = SEM(ch_in,reduction=reduction)
        self.sem2 = SEM(ch_in,reduction=reduction)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU(True)
                
    def forward(self, x):
        x1 = self.squeeze(x)
        
        x1 = self.c1(x1) + x1
        x1 = self.out(self.dropout(torch.cat([x1, self.c2(x1)],1)))
        x2 = self.sem1(x1)
        x = self.sem2(x)
        
        return self.relu(x + x1 + x2) 
    
    
# Object Boundary Attention
class OBAtten(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, reduction):
        super(OBAtten, self).__init__()
        
        self.FE = FeatureExtraction(ch_in)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//2, kernel_size=1,groups=(ch_out//2)//2,bias=False),
            nn.GroupNorm(2, ch_out//2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out//2, ch_out, kernel_size=1,groups=(ch_out//2),bias=False),
            nn.GroupNorm(2, ch_out)
            )
        
        self.sem = SEM(ch_out,reduction=reduction)
        self.relu = nn.ReLU(True)
        
    def forward(self, x):
        feat = self.conv2(self.conv1(self.FE.EF(x, threshold=[0.05,0.2], k_idx=0)))
        feat1 = self.sem(feat)
        
        return self.relu(x + feat + feat1)
    
    
# Multiscale Self-attention Attention   
class MSAtten(nn.Module):
    def __init__(self, ch_in, head, ps, rr, reduction, aa, dropout):
        super(MSAtten, self).__init__()
        self.c = ch_in//rr
        self.H = head
        self.ps = ps
        
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)
        
        self.squeeze = iConv(ch_in, self.c, 1, self.c)
        
        self.q = iConv(self.c, self.c*head, 3, self.c//4)
        self.k = iConv(self.c, self.c*head, 1, self.c//2)
        self.v = iConv(self.c, self.c*head, 3, self.c//2)
        
        self.mamba = Mamba(self.c*head,self.c*head,(self.ps*2)**2,self.c*head) if aa else nn.Identity()
        
        self.obatten = OBAtten(self.c*head,self.c*head,kernel=3,reduction=reduction) if aa else nn.Identity()
        
        self.out = nn.Sequential(
            nn.Conv2d((self.c*head), ch_in, kernel_size=1,groups=(self.c*head)//2,bias=False),
            nn.GroupNorm(2, ch_in)
            )
                
    # Channel Self-attention
    def CSA(self, x):
        _, _, h, _ = x.size()
        
        q = rearrange(self.q(x), 'b (c H) h w -> b H c (h w)', H=self.H)
        k = rearrange(self.k(x), 'b (c H) h w -> b H c (h w)', H=self.H)
        v = rearrange(self.v(x), 'b (c H) h w -> b H c (h w)', H=self.H)
        
        _, _, c, _ = k.shape
        q = torch.matmul(q, k.transpose(-1, -2)) * (c ** -0.5)
        
        q = self.dropout(self.softmax(q))
        v = torch.matmul(q, v)
        
        v = rearrange(v, 'b H c (h w) -> b (c H) h w', h=h)
        return v
    
    def Mamba(self, x):
        return x * self.dropout(self.softmax(self.mamba(x)))
    
    def OBA(self, x):
        return x * self.dropout(self.softmax(self.obatten(x)))
    
    def forward(self, x):
        x1 = self.squeeze(x)
        
        x1 = self.CSA(x1)
        x2 = self.Mamba(x1) + self.OBA(x1)
        return x * self.out(x1 + x2)
        
        
class CoordAtten(nn.Module):
    def __init__(self, inp, oup, dropout, reduction=32):
        super(CoordAtten, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d(output_size=(None, 1)) # X Avg Pool
        self.pool_w = nn.AdaptiveAvgPool2d(output_size=(1, None)) # Y Avg Pool
        
        mip = max(8, inp // reduction)
        
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.gn = nn.GroupNorm(2, mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.gn(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        
        out = x * self.dropout(a_w * a_h)
        
        return out
        
# Multi Attention Module       
class MAM(nn.Module):
    def __init__(self, ch_in, ch_out, ps, reduction, aa, dropout):
        super(MAM, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, groups=ch_out//2, bias=True),
            nn.GroupNorm(2, ch_out)
            )
        self.MeanAtten = nn.Sequential(
            nn.Conv2d(1, ch_out, kernel_size=1, padding=0, groups=1, bias=False),
            nn.Softmax(-1),
            nn.Dropout(dropout)
            )
        self.CoordAtten = CoordAtten(ch_out, ch_out, dropout)
        self.MSAtten = MSAtten(ch_out, 1, ps, 8, reduction, aa, dropout)
        
        self.relu = nn.ReLU(True)
    
    def forward(self, x, ef):            
        x = self.squeeze(self.relu(x))
        
        meanatten = x * self.MeanAtten(x.mean(dim=1, keepdim=True))
        coordatten = self.CoordAtten(x)
        msatten = self.MSAtten(x)
        
        return (meanatten + coordatten + msatten)/3

        
class Decoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Decoder, self).__init__()
        
        self.Atten2 = MAM(ch_in[1]*2, 64, 4, 2, True, 0.025)
        self.Atten3 = MAM(ch_in[2]*2, 64, 4, 2, False, 0.0125)
        self.Atten4 = MAM(ch_in[3]*2, 64, 2, 2, False, 0.)
        
        self.layers1 = nn.ModuleList([])
        self.layers1.append(nn.ModuleList([
            iConv(64+ch_in[0], 64, 3, 64//2),
            FDM(64, 16, 16)
            ]))
        
        self.layers2 = nn.ModuleList([])
        self.layers2.append(nn.ModuleList([
            iConv(64+ch_in[1], 64, 3, 64//2),
            FDM(64, 16, 16)
            ]))
        
        self.layers3 = nn.ModuleList([])
        self.layers3.append(nn.ModuleList([
            iConv(64+ch_in[2], 64, 3, 64//2),
            FDM(64, 16, 16)
            ]))
        
        self.layers4 = nn.ModuleList([])
        self.layers4.append(nn.ModuleList([
            iConv(128+ch_in[3], 64, 1, 64//2),
            FDM(64, 16, 16)
            ]))
        
        self.layers5 = nn.ModuleList([])
        self.layers5.append(nn.ModuleList([
            iConv(ch_in[4], 128, 1, 128//2),
            FDM(128, 32, 32)
            ]))
        
        self.out = nn.Sequential(
                iConv(64, 64, 3, 64//4, True),
                nn.Conv2d(64, 32, kernel_size=1,bias=False),
                nn.GroupNorm(2, 32),
                nn.Conv2d(32, ch_out, kernel_size=1,bias=False)
                )
        
    def forward(self, x1, x2, x3, x4, x5, m2, m3, m4, ef, img_shape):
        m4 = (x4 - m4)**2
        m3 = (x3 - m3)**2
        m2 = (x2 - m2)**2
        
        ef4 = (m4.mean(dim=1, keepdim=True) + F.interpolate(ef, size=m4.shape[2:], mode='bilinear', align_corners=True))
        ef3 = (m3.mean(dim=1, keepdim=True) + F.interpolate(ef, size=m3.shape[2:], mode='bilinear', align_corners=True)) * F.interpolate(ef4, scale_factor=(2), mode='bilinear', align_corners=True)
        ef2 = (m2.mean(dim=1, keepdim=True) + F.interpolate(ef, size=m2.shape[2:], mode='bilinear', align_corners=True)) * F.interpolate(ef3, scale_factor=(2), mode='bilinear', align_corners=True)
        
        atten4 = self.Atten4(torch.cat([x4, m4],1), ef4)
        atten3 = self.Atten3(torch.cat([x3, m3],1), ef3) + F.interpolate(atten4, scale_factor=(2), mode='bilinear')
        atten2 = self.Atten2(torch.cat([x2, m2],1), ef2) + F.interpolate(atten3, scale_factor=(2), mode='bilinear')

        for (conv, FDM) in self.layers5:
            x = FDM(conv(x5))
            
        x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=True)
        x4 = torch.cat([x,x4],1)
        for (conv, FDM) in self.layers4:
            x = FDM(conv(x4)) * atten4
            
        x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=True)
        x3 = torch.cat([x,x3],1)
        for (conv, FDM) in self.layers3:
            x = FDM(conv(x3)) * atten3
            
        x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=True)
        x2 = torch.cat([x,x2],1)
        for (conv, FDM) in self.layers2:
            x = FDM(conv(x2)) * atten2
            
        x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=True)
        x1 = torch.cat([x,x1],1)
        for (conv, FDM) in self.layers1:
            x = FDM(conv(x1))
            
        x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=True)
        
        return self.out(x)
   

class UL_IAD(nn.Module):
    def __init__(self, ch_in, segout, training_mode):
        super(UL_IAD, self).__init__()
        self.training_mode = training_mode
        
        if training_mode == 'FSS i':
            self.FFM_Net = FFM_Net(3, segout, 256, 5)
            
        elif training_mode == 'FSS l':
            self.image_memory = torch.Tensor()
            self.image_memory.requires_grad = False
            
            E = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.E = create_feature_extractor(E, return_nodes=['conv1', 'layer1', 'layer2', 'layer3', 'layer4'])
            for p in self.E.layer1.parameters(): p.requires_grad = False
            for p in self.E.layer2.parameters(): p.requires_grad = False
            for p in self.E.layer3.parameters(): p.requires_grad = False
            
            self.D = Decoder(ch_in, segout+1)
    
    def forward(self, x, ffm=None):
        SR = 0.
        if self.training_mode == 'FSS i':
            SR = self.FFM_Net(x)
            
        elif self.training_mode == 'FSS l':
            img_shape = x.shape[2:]
            feats = self.E(x)
            mem_feats = self.E(self.image_memory.to(x.device))
            x1, x2, x3, x4, x5 = feats.values()
            _ , m2, m3, m4, _  = mem_feats.values()
            
            x = T.Grayscale(num_output_channels=1)(x)
            mem = T.Grayscale(num_output_channels=1)(self.image_memory.to(x.device))
            ix = T.RandomInvert(p=1.)(x)
            iffm = T.RandomInvert(p=1.)(ffm)
            imem = T.RandomInvert(p=1.)(mem)
            dx = (x - mem)**2
            idx = (ix - imem)**2
            dffm = (ffm - mem)**2
            idffm = (iffm - imem)**2
            ef = dx * idx * dffm * idffm
            
            SR = self.D(x1, x2, x3, x4, x5, m2, m3, m4, ef, img_shape)
            
        return SR


class Learner(object):
    def __init__(self, cfg, model, AM, optimizer, loss_eq):
        
        self.cfg = cfg
        self.model = model.to(self.cfg.cuda)
        self.AM = AM.to(self.cfg.cuda) if AM is not None else model.to(self.cfg.cuda)
        self.optimizer = optimizer
        self.MAELoss = loss_eq[0].to(self.cfg.cuda)
        self.FocalLoss = loss_eq[1].to(self.cfg.cuda)
        self.image_memory = torch.Tensor()
                
    def loss(self, R, GT, training_mode):
        if training_mode == 'FSS i':
            total_loss = self.MAELoss(R,GT)
        elif training_mode == 'FSS l':
            loss_factor = random.uniform(.6, .8)
            loss1 = self.MAELoss(R[:,1:,:,:],GT)
            loss2 = self.FocalLoss(R,GT)
            total_loss = (loss_factor*loss1) + ((1-loss_factor)*loss2)
        
        return total_loss
    
    def learn(self, image, GT, training_mode, mode='test'):
        
        self.training_mode = training_mode
        image = image.to(self.cfg.cuda)
        GT = [GT[0].to(self.cfg.cuda),GT[1].to(self.cfg.cuda)]
        
        if mode == 'train':
            self.model.train(True)
            self.AM.train(False)
            
            if self.training_mode == 'FSS i':
                GT = [GT[0].to(self.cfg.cuda),T.Grayscale(num_output_channels=1)(image)+GT[1].to(self.cfg.cuda)]
                
                R = self.model(image).contiguous()
                
                loss = self.loss(R, GT[1], self.training_mode)
                
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            elif self.training_mode == 'FSS l':
                self.model.image_memory = torch.Tensor()
                if sum(self.image_memory.shape) == 0:
                    self.image_memory = torch.zeros(image.shape).to(image.device)
                    
                for b1 in range(image.shape[0]):
                    if GT[0][b1] == 0:
                        if self.image_memory.shape[0] == self.cfg.batch_size:
                            self.image_memory = self.image_memory[1:,:].to(image.device)
                        self.image_memory = torch.cat([self.image_memory.to(image.device),image[b1].unsqueeze(0)],0)
                    
                    l = []
                    for b2 in range(self.image_memory.shape[0]):
                        l.append(F.mse_loss(image[b1], self.image_memory[b2], reduction='mean'))
                        
                    self.model.image_memory = torch.cat([self.model.image_memory.to(image.device),self.image_memory[l.index(min(l)),:].unsqueeze(0)],0)
                
                with torch.no_grad():
                    R1 = self.AM(image)
                    
                R = self.model(image, R1).contiguous()
                
                loss = self.loss(R, GT[1], self.training_mode)
                
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        else:
            self.model.train(False)
            self.AM.train(False)
            
            with torch.no_grad():
                if self.training_mode == 'FSS i':
                    GT = [GT[0].to(self.cfg.cuda),T.Grayscale(num_output_channels=1)(image)+GT[1].to(self.cfg.cuda)]
                    
                    R = self.model(image).contiguous()
                    
                    loss = self.loss(R, GT[1], self.training_mode)
                    
                elif self.training_mode == 'FSS l':
                    R1 = self.AM(image)
                    
                    R = self.model(image, R1).contiguous()
                    
                    loss = self.loss(R, GT[1], self.training_mode)
                    
        return [R[:,1:,:,:], GT, loss]
        
        