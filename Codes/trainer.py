# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:04:24 2023

@author: Omar Al-maqtari
"""

import os
import time
from datetime import datetime

import torch
from torch import nn
from torch import optim

from UL_IAD import Learner, UL_IAD

import csv
from evaluation import Evaluation
from loss import displayfigures
from focal_loss import FocalLoss
from scheduler import CosineAnnealingWarmupRestarts
import matplotlib.pyplot as plt
    

class Trainer(object):
    def __init__(self, cfg, train_loader, valid_loader, test_loader):
        # Config
        self.cfg = cfg
        
        # Paths
        self.model_path = cfg.model_path
        self.result_path = cfg.result_path
        self.SR_path = cfg.SR_path
        self.net_path = os.path.join(self.model_path, cfg.report_name+'.pkl')
        self.AM_path = os.path.join(self.model_path, 'FSS i_'+ self.cfg.model_type +'_'+ (self.cfg.dataset.replace('MVTec_','')) +'.pkl') if self.cfg.training_mode == 'FSS l' else None
        
        # Report file
        self.report_name = cfg.report_name
        self.report = open(self.result_path+self.report_name+'.txt','a+')
        self.report.write('\n'+str(datetime.now()))
        self.report.write('\n'+str(cfg))
        
		# Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

		# Hyper-parameters
        self.lr = cfg.lr
        self.beta1 = cfg.beta1
        self.beta2 = cfg.beta2
        self.num_epochs_decay = cfg.num_epochs_decay
        self.aug_prob = cfg.aug_prob

		# Training settings
        self.mode = cfg.mode
        self.training_mode = cfg.training_mode
        self.img_ch = cfg.img_ch
        self.out_ch = cfg.out_ch
        self.img_size = cfg.image_height
        self.num_epochs = cfg.num_epochs
        self.batch_size = cfg.batch_size
        
        # Models
        print("initialize model...")
        self.model = None
        self.optimizer = None
        self.model_type = cfg.model_type
        self.dataset = cfg.dataset
        self.device = torch.device(cfg.cuda if torch.cuda.is_available() else 'cpu')
        self.loss_eq = []
        self.loss_eq.append(nn.L1Loss())
        self.loss_eq.append(FocalLoss(gamma=4))
        
        # EMT+
        if self.model_type == 'UL_IAD':
            ch_in = [64, 64, 128, 256, 512]
            
            self.model = UL_IAD(ch_in, self.out_ch, self.training_mode)
            
            self.AM = UL_IAD(ch_in, self.out_ch, 'FSS i') if self.training_mode == 'FSS l' else None
            
        self.optimizer = optim.Adam(self.model.parameters(), self.lr, [self.beta1, self.beta2], weight_decay=4e-4)
        self.lr_sch = CosineAnnealingWarmupRestarts(self.optimizer, first_cycle_steps=self.num_epochs, max_lr=self.lr, min_lr=0.0001, warmup_steps=int(self.num_epochs*0.1))
        
        if self.mode == 'train':
            self.print_model(self.model, self.model_type)
            
    def print_model(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        
        print(name)
        self.report.write('\n'+str(name))
        print(self.training_mode)
        self.report.write('\n'+str(self.training_mode))
        print("The number of parameters: {}".format(num_params))
        self.report.write("\n The number of parameters: {}".format(num_params))
        self.report.write('\n'+str(model))
        
        
    def train(self):
		#====================================== Training ===========================================#
		#===========================================================================================#
        model_score = 0.
        t = time.time()
        elapsed = 0.# Time of inference
        
		# Model Train
        if self.training_mode == 'FSS l' and os.path.isfile(self.AM_path):
            self.AM = torch.load(self.AM_path, map_location='cpu').to(self.device)
            print('%s is Successfully Loaded from %s'%(self.model_type,self.AM_path))
            self.report.write('\n%s is Successfully Loaded from %s'%(self.model_type,self.AM_path))
            
        if os.path.isfile(self.net_path):
            
            Train_results = open(self.result_path+self.report_name+'_Train_result.csv', 'a', encoding='utf-8', newline='')
            twr = csv.writer(Train_results)
            
            Valid_results = open(self.result_path+self.report_name+'_Valid_result.csv', 'a', encoding='utf-8', newline='')
            vwr = csv.writer(Valid_results)
            
        else:
            Train_results = open(self.result_path+self.report_name+'_Train_result.csv', 'a', encoding='utf-8', newline='')
            twr = csv.writer(Train_results)
            
            Valid_results = open(self.result_path+self.report_name+'_Valid_result.csv', 'a', encoding='utf-8', newline='')
            vwr = csv.writer(Valid_results)
            
            twr.writerow(['Train_model','Model_type','Dataset','LR','Epochs','Augmentation_prob'])
            twr.writerow([self.report_name,self.model_type,self.dataset,self.lr,self.num_epochs,self.aug_prob])
            if self.training_mode != 'FSS i':
                twr.writerow(['Epoch','LR','Loss','Acc','RC','PC','F1','OIS','IoU','AIU','mIoU','PRC','ROC_I','ROC_P','DC'])
            else:
                twr.writerow(['Epoch','LR','Loss','MAE','MSE'])
            vwr.writerow(['Train_model','Model_type','Dataset','LR','Epochs','Augmentation_prob'])
            vwr.writerow([self.report_name,self.model_type,self.dataset,self.lr,self.num_epochs,self.aug_prob])
            if self.training_mode != 'FSS i':
                vwr.writerow(['Epoch','LR','Loss','Acc','RC','PC','F1','OIS','IoU','AIU','mIoU','PRC','ROC_I','ROC_P','DC'])
            else:
                vwr.writerow(['Epoch','LR','Loss','MAE','MSE'])
                
        # Training
        self.Learning_model = Learner(self.cfg, self.model, self.AM, self.optimizer, self.loss_eq)
        if self.training_mode != 'FSS i':
            results = [["Loss",[],[]],["Acc",[],[]],["RC",[],[]],["PC",[],[]],["F1",[],[]],["OIS",[],[]],["IoU",[],[]],["AIU",[],[]],["mIoU",[],[]],["PRC",[],[]],["ROC_I",[],[]],["ROC_P",[],[]],["DC",[],[]]]
        else:
            results = [["Loss",[],[]],["MAE",[],[]],["MSE",[],[]]]
             
        for epoch in range(self.num_epochs):
            evaluator = Evaluation()
            
            for i, (image, GT, name) in enumerate(self.train_loader):
                R = self.Learning_model.learn(image, GT, self.training_mode, mode='train')
                
                # Get metrices results
                metrics = evaluator.metrics(R[0], R[1], R[2], self.training_mode)
                
            metavg = evaluator.metrics_avg(metrics, self.training_mode)
            
            for i in range(len(results)):
                results[i][1].append((metavg[i]))
            
            # Print the report info
            print('\n\nEpoch [%d/%d], LR: [%0.5f] \n[Training]' % (epoch+1, self.num_epochs, self.lr))
            self.report.write('\n\nEpoch [%d/%d], LR: [%0.5f] \n[Training]' % (epoch+1, self.num_epochs, self.lr))
            if self.training_mode != 'FSS i':
                print('\n[R] Loss: %.4f, Acc: %.4f, RC: %.4f, PC: %.4f, F1: %.4f, OIS: %.4f, IoU: %.4f, AIU: %.4f, mIoU: %.4f, PRC: %.4f, ROC_I: %.4f, ROC_P: %.4f, DC: %.4f' % (
                    metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6], metavg[7], metavg[8], metavg[9], metavg[10], metavg[11], metavg[12]))
                self.report.write('\n[R] Loss: %.4f, Acc: %.4f, RC: %.4f, PC: %.4f, F1: %.4f, OIS: %.4f, IoU: %.4f, AIU: %.4f, mIoU: %.4f, PRC: %.4f, ROC_I: %.4f, ROC_P: %.4f, DC: %.4f' % (
                    metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6], metavg[7], metavg[8], metavg[9], metavg[10], metavg[11], metavg[12]))
                twr.writerow([epoch+1, self.lr, metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6], metavg[7], metavg[8], metavg[9], metavg[10], metavg[11], metavg[12]])
            else:
                print('\n[R] Loss: %.4f, MAE: %.4f, MSE: %.4f' % (metavg[0], metavg[1], metavg[2]))
                self.report.write('\n[R] Loss: %.4f, MAE: %.4f, MSE: %.4f' % (metavg[0], metavg[1], metavg[2]))
                twr.writerow([epoch+1, self.lr, metavg[0], metavg[1], metavg[2]])
                
    		# Clear unoccupied GPU memory after each epoch
            torch.cuda.empty_cache()
            
            #========================== Validation ====================================#
            
            evaluator = Evaluation()
            
            for i, (image, GT, name) in enumerate(self.valid_loader):
                R = self.Learning_model.learn(image, GT, self.training_mode, mode='valid')
                
                # Get metrices results
                metrics = evaluator.metrics(R[0], R[1], R[2], self.training_mode)
                
            metavg = evaluator.metrics_avg(metrics, self.training_mode)
            
            for i in range(len(results)):
                results[i][2].append((metavg[i]))
            
            # Print the report info
            print('\n[Validation]')
            self.report.write('\n[Validation]')
            if self.training_mode != 'FSS i':
                print('\n[R] Loss: %.4f, Acc: %.4f, RC: %.4f, PC: %.4f, F1: %.4f, OIS: %.4f, IoU: %.4f, AIU: %.4f, mIoU: %.4f, PRC: %.4f, ROC_I: %.4f, ROC_P: %.4f, DC: %.4f' % (
                    metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6], metavg[7], metavg[8], metavg[9], metavg[10], metavg[11], metavg[12]))
                self.report.write('\n[R] Loss: %.4f, Acc: %.4f, RC: %.4f, PC: %.4f, F1: %.4f, OIS: %.4f, IoU: %.4f, AIU: %.4f, mIoU: %.4f, PRC: %.4f, ROC_I: %.4f, ROC_P: %.4f, DC: %.4f' % (
                    metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6], metavg[7], metavg[8], metavg[9], metavg[10], metavg[11], metavg[12]))
                vwr.writerow([epoch+1, self.lr, metavg[0], metavg[1], metavg[2], metavg[3], metavg[4], metavg[5], metavg[6], metavg[7], metavg[8], metavg[9], metavg[10], metavg[11], metavg[12]])
            else:
                print('\n[R] Loss: %.4f, MAE: %.4f, MSE: %.4f' % (metavg[0], metavg[1], metavg[2]))
                self.report.write('\n[R] Loss: %.4f, MAE: %.4f, MSE: %.4f' % (metavg[0], metavg[1], metavg[2]))
                vwr.writerow([epoch+1, self.lr, metavg[0], metavg[1], metavg[2]])
            
            # Decay learning rate
            self.lr_sch.step()
            self.lr = self.lr_sch.get_lr()[0]
            
            # Save Best Model
            if self.training_mode == 'FSS i':
                if metavg[2] > model_score:
                    model_score = metavg[2]
                    print('\nBest %s model score : %.4f'%(self.model_type,model_score))
                    self.report.write('\nBest %s model score : %.4f'%(self.model_type,model_score))
                    torch.save(self.model,self.net_path)
            elif self.training_mode == 'FSS l':
                if metavg[11] > model_score:
                    model_score = metavg[11]
                    print('\nBest %s model score : %.4f'%(self.model_type,model_score))
                    self.report.write('\nBest %s model score : %.4f'%(self.model_type,model_score))
                    torch.save(self.model,self.net_path)
                    
            # Clear unoccupied GPU memory after each epoch
            torch.cuda.empty_cache()
            
        displayfigures(results, self.result_path, self.report_name)
        
        Train_results.close()
        Valid_results.close()
        elapsed = time.time() - t
        print("\nElapsed time: %f seconds.\n\n" %elapsed)
        self.report.write("\nElapsed time: %f seconds.\n\n" %elapsed)
        self.report.close()
        
                    
    def test(self):		
		#===================================== Test ====================================#
        
        # Load Trained Model
        if self.training_mode == 'FSS i' and os.path.isfile(self.net_path):
            self.model = torch.load(self.net_path,map_location='cpu').to(self.device)
            print('%s is Successfully Loaded from %s'%(self.model_type,self.net_path))
            self.report.write('\n%s is Successfully Loaded from %s'%(self.model_type,self.net_path))
        
        elif self.training_mode == 'FSS l' and os.path.isfile(self.net_path) and os.path.isfile(self.AM_path):
            self.model = torch.load(self.net_path,map_location='cpu').to(self.device)
            print('%s is Successfully Loaded from %s'%(self.model_type,self.net_path))
            self.report.write('\n%s is Successfully Loaded from %s'%(self.model_type,self.net_path))
            
            self.AM = torch.load(self.AM_path,map_location='cpu').to(self.device)
            print('%s is Successfully Loaded from %s'%(self.model_type,self.AM_path))
            self.report.write('\n%s is Successfully Loaded from %s'%(self.model_type,self.AM_path))
            
        else: 
            print("Trained model or AM NOT found, Please train models first")
            self.report.write("\nTrained model or AM NOT found, Please train models first")
            return
        
        self.Learning_model = Learner(self.cfg, self.model, self.AM, self.optimizer, self.loss_eq)
        
        if self.training_mode != 'FSS i':
            results = [["Loss",[],[]],["Acc",[],[]],["RC",[],[]],["PC",[],[]],["F1",[],[]],["OIS",[],[]],["IoU",[],[]],["AIU",[],[]],["mIoU",[],[]],["PRC",[],[]],["ROC_I",[],[]],["ROC_P",[],[]],["DC",[],[]]]
        else:
            results = [["Loss",[],[]],["MAE",[],[]],["MSE",[],[]]]
            
        evaluator = Evaluation()
        elapsed = 0.# Time of inference
        RC_curve = 0.
        PC_curve = 0.
        RC_all = []
        PC_all = []
        
        for i, (image, GT, name) in enumerate(self.test_loader):
            #Time of inference
            t = time.time()
            
            R = self.Learning_model.learn(image, GT, self.training_mode, mode='test')
            
            elapsed = (time.time() - t)
            
            # Get metrices results
            metrics = evaluator.metrics(R[0], R[1], R[2], self.training_mode)
            if self.training_mode != 'FSS i': 
                RC_all.append(metrics[11])
                PC_all.append(metrics[12])
            '''    
            img = plt.figure(frameon=False)
            ax = plt.Axes(img, [0., 0., 1., 1.])
            ax.set_axis_off()
            img.add_axes(ax)
            plt.imshow(image[0,:,:,:].permute(1,2,0).cpu().numpy())
            #img.savefig(self.SR_path+name[i], bbox_inches='tight',transparent=True, pad_inches=0)
            
            img = plt.figure(frameon=False)
            ax = plt.Axes(img, [0., 0., 1., 1.])
            ax.set_axis_off()
            img.add_axes(ax)
            plt.imshow(R[0][0,0,:,:].detach().cpu().numpy(), cmap='gray')
            #img.savefig(self.SR_path+self.model_type+' '+name[i], dpi=80, bbox_inches='tight',transparent=True, pad_inches=0)
            if i == 5:
                print(i)
                return
            '''
        metavg = evaluator.metrics_avg(metrics, self.training_mode)
        
        for i in range(len(results)):
            results[i][1].append((metavg[i]))
            
        elapsed = elapsed/(R[0].size(0))
        
        f = open(os.path.join(self.result_path,'Test_result.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        if self.training_mode != 'FSS i':
            wr.writerow(['Report_file','Model_type','Dataset','Loss','Acc','RC','PC','F1','OIS','IoU','AIU','mIoU','PRC','ROC_I','ROC_P','DC','Time of inference','LR','Epochs','Augmentation_prob'])
            wr.writerow([self.report_name,self.model_type,self.dataset,metavg[0],metavg[1],metavg[2],metavg[3],metavg[4],metavg[5],metavg[6],metavg[7],metavg[8],metavg[9],metavg[10],metavg[11],metavg[12],elapsed,self.lr,self.num_epochs,self.aug_prob])
        else:
            wr.writerow(['Report_file','Model_type','Dataset','Loss','MAE','MSE','Time of inference','LR','Epochs','Augmentation_prob'])
            wr.writerow([self.report_name,self.model_type,self.dataset,metavg[0],metavg[1],metavg[2],elapsed,self.lr,self.num_epochs,self.aug_prob])
            
        f.close()
        
        print('Results have been Saved')
        self.report.write('\nResults have been Saved\n\n')
        
        # Clear unoccupied GPU memory after each epoch
        torch.cuda.empty_cache()
        
        self.report.close()
        