# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:04:24 2023

@author: Omar Al-maqtari
"""

import os
import argparse
from trainer import Trainer
from data_loader import get_loader
import torch
from torch.backends import cudnn
from anomaly_mask_cfg import anomaly_mask_cfg

def main(cfg):
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if cfg.model_type not in ['UL_IAD']:
        print('ERROR!! model_type should be selected in UL_IAD')
        print('Your input for model_type was %s'%cfg.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)
    if not os.path.exists(cfg.result_path):
        os.makedirs(cfg.result_path)
        cfg.result_path = os.path.join(cfg.result_path,cfg.model_type)
    
    print(cfg)

    train_loader = get_loader(cfg, mode='train', aug_prob=cfg.aug_prob)
    valid_loader = get_loader(cfg, mode='test', aug_prob=0.)
    test_loader = get_loader(cfg, mode='test', aug_prob=0.)
    
    trainer = Trainer(cfg, train_loader, valid_loader, test_loader)
    
    # Train and sample the images
    if cfg.mode == 'train':
        trainer.train()
    elif cfg.mode == 'test':
        trainer.test()
    
        
if __name__ == '__main__':
    amc = anomaly_mask_cfg()
    for training_mode in ['FSS i', 'FSS l']:# 'FSS i', 'FSS l'
        for model_type in ['UL_IAD']:
            for dataset in ['MVTec', 'DAGM', 'BTD']:# 'MVTec', 'DAGM', 'BTD'
                for subdataset in ['Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']:
                # MVTec: bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper'
                # DAGM: 'Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10'
                # BTD: '01','02','03'
                    for mode in ['train', 'test']:# 'train', 'test'
                        parser = argparse.ArgumentParser()
                        if training_mode == 'FSS i':
                            epochs = 200
                            num_epochs_decay = 6
                            batch_size = 12
                            aug_prob = 0.2
                        elif training_mode == 'FSS l':
                            epochs = 300
                            num_epochs_decay = 8
                            batch_size = 12
                            aug_prob = .5
                            
                        # model hyper-parameters
                        parser.add_argument('--img_ch', type=int, default=3)
                        parser.add_argument('--out_ch', type=int, default=1)
                        parser.add_argument('--image_height', type=int, default=256)
                        parser.add_argument('--image_width', type=int, default=256)
                        parser.add_argument('--num_workers', type=int, default=0)
                        
                        # training hyper-parameters
                        parser.add_argument('--lr', type=float, default=0.005)
                        parser.add_argument('--num_epochs', type=int, default=epochs)
                        parser.add_argument('--num_epochs_decay', type=int, default=num_epochs_decay)
                        parser.add_argument('--batch_size', type=int, default=batch_size)
                        parser.add_argument('--loss_threshold', type=float, default=0.5)
                        parser.add_argument('--beta1', type=float, default=0.9)       # momentum1 in Adam or SGD
                        parser.add_argument('--beta2', type=float, default=0.999)     # momentum2 in Adam
                        parser.add_argument('--aug_prob', type=float, default=aug_prob)
                        parser.add_argument('--use_mask', type=float, default=amc[subdataset]['use_mask'])
                        parser.add_argument('--bg_threshold', type=float, default=amc[subdataset]['bg_threshold'])
                        parser.add_argument('--bg_reverse', type=float, default=amc[subdataset]['bg_reverse'])
                        
                        # misc
                        parser.add_argument('--mode', type=str, default=mode, help='train, test')
                        parser.add_argument('--training_mode', type=str, default=training_mode, help='FSS i,FSS l')
                        parser.add_argument('--report_name', type=str, default= training_mode +'_'+ model_type +'_'+ dataset +'_'+ subdataset)
                        parser.add_argument('--dataset', type=str, default=dataset+'_'+subdataset)
                        parser.add_argument('--model_type', type=str, default=model_type, help='EMT+')
                        parser.add_argument('--model_path', type=str, default='.../model/')
                        parser.add_argument('--result_path', type=str, default='.../results/')
                        parser.add_argument('--dataset_path', type=str, default='/.../'+dataset+'/'+subdataset+'/')
                        parser.add_argument('--SR_path', type=str, default='/.../'+dataset+'/'+subdataset+'/SR/')
                        
                        parser.add_argument('--cuda', type=str, default='cuda:0')
                        
                        cfg = parser.parse_args()
                        main(cfg)        