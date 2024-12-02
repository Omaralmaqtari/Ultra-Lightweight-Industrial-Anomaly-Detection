# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:04:24 2023

@author: Omar Al-maqtari
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# R : Model Result
# GT : Ground Truth

class Evaluation(object):
    def __init__(self):
        
        self.thresholdlist = np.linspace(0, 1, 51)
        self.results = []
        self.loss = 0.
        self.Acc = 0.	# Accuracy
        self.RC = 0.	# Recall (Sensitivity)
        self.PC = 0. 	# Precision
        self.OIS = 0.   # 
        self.IoU = 0.   # Intersection over Union (Jaccard Index)
        self.AIU = 0.   #
        self.mIoU = 0.	# mean of Intersection over Union (mIoU)
        self.R_class = [] #
        self.R_mask = [] #
        self.GT_class = []
        self.GT_mask = []
        self.DC = 0.	# Dice Coefficient
        self.length = 0
        self.RC_all = 0
        self.PC_all = 0
        self.MAE = 0.   # Mean Absolute Error 
        self.MSE = 0.   # Mean Square Error
            
    def get_results(self, R, GT):
        Acc =0.
        Accb = 0.
        Acc_all = []
        RC = 0.
        RCb = 0.
        RC_all = []
        PC = 0.
        PCb = 0.
        PC_all = []
        F1 = 0.
        OIS = 0.
        F1_all = []
        IoUf = 0.
        AIU = 0.
        mIoU = 0.
        AmIoU = 0.
        IoUf_all = []
        mIoU_all = []
        DC = 0.
        DCb = 0.
        DC_all = []
        GT_copy = copy.deepcopy(GT)
        
        for threshold in self.thresholdlist:
            R_copy = copy.deepcopy(R)
            R_copy[R_copy < threshold] = 0.
            R_copy[R_copy >= threshold] = 1.
            
            # TP: True Positive
            # TN: True Negative
            # FP: False Positive
            # FN: False Negative
            
            TP = torch.sum((R_copy==1)&(GT_copy==1))
            TN = torch.sum((R_copy==0)&(GT_copy==0))
            FP = torch.sum((R_copy==1)&(GT_copy==0))
            FN = torch.sum((R_copy==0)&(GT_copy==1))
            
            # Accuracy
            corr = R_copy.eq(GT_copy).sum()
            tensor_size = 1.
            for i in R_copy.shape:
                tensor_size *= i
            Acc_copy = float(corr)/float(tensor_size)
            Acc_all.append(Acc_copy)
            
            # Recall == Sensitivity
            RC_copy = float(TP)/(float(TP+FN) + 1e-12)
            RC_all.append(RC_copy)
            
            # Precision
            PC_copy = float(TP)/(float(TP+FP) + 1e-12)
            PC_all.append(PC_copy)
            
            # F1-Score == Dice Score
            F1_copy = (2*RC_copy*PC_copy)/(RC_copy+PC_copy + 1e-12)
            F1_all.append(F1_copy)
            
            # IoU of Foreground
            Union1 = (torch.sum(R_copy==1) + torch.sum(GT_copy==1)) - TP
            IoUf_copy = float(TP)/(float(Union1) + 1e-12)
            IoUf_all.append(IoUf_copy)
            
            # IoU of Background
            Union2 = (torch.sum(R_copy==0) + torch.sum(GT_copy==0)) - TN
            IoUb = float(TN)/(float(Union2) + 1e-12)
            
            # mIoU
            mIoU_copy = (IoUf_copy + IoUb) / 2
            mIoU_all.append(mIoU_copy)
            
            # DC
            Union = torch.sum(R_copy==1)+torch.sum(GT_copy==1)
            DC_copy = float(2*TP)/(float(Union) + 1e-12)
            DC_all.append(DC_copy)
            
            if threshold == 0.5:
                Acc = copy.deepcopy(Acc_copy)
                RC = copy.deepcopy(RC_copy)
                PC = copy.deepcopy(PC_copy)
                F1 = copy.deepcopy(F1_copy)
                IoUf = copy.deepcopy(IoUf_copy)
                mIoU = copy.deepcopy(mIoU_copy)
                DC = copy.deepcopy(DC_copy)
                
            if Acc_copy > Accb:
                Accb = copy.deepcopy(Acc_copy)
                
            if RC_copy > RCb:
                RCb = copy.deepcopy(RC_copy)
                
            if PC_copy > PCb:
                PCb = copy.deepcopy(PC_copy)
                
            if F1_copy > OIS:    
                OIS = copy.deepcopy(F1_copy)
                
            if IoUf_copy > AIU:
                AIU = copy.deepcopy(IoUf_copy)
                
            if mIoU_copy > AmIoU:
                AmIoU = copy.deepcopy(mIoU_copy)
                
            if DC_copy > DCb:
                DCb = copy.deepcopy(DC_copy)
            
        return [Acc, Accb, Acc_all, RC, RCb, RC_all, PC, PCb, PC_all, F1, OIS, F1_all, IoUf, AIU, \
    IoUf_all, mIoU, AmIoU, mIoU_all, DC, DCb, DC_all]

    def metrics(self, R, GT, total_loss, training_mode):
        self.loss += total_loss.detach().item()
        if training_mode != 'FSS i':
            self.results = self.get_results(R.view(-1).detach(), GT[1].view(-1).detach())
            self.Acc += self.results[0]
            self.RC += self.results[3]
            self.RC_all = self.results[5]
            self.PC += self.results[6]
            self.PC_all = self.results[8]
            self.OIS += self.results[10]
            self.IoU += self.results[12]
            self.AIU += self.results[13]
            self.mIoU += self.results[15]
            self.DC += self.results[18]
            R_class = torch.topk(R.flatten(1), 100)[0].mean(dim=1)
            GT_class = GT[0]
            self.R_class.extend(R_class.cpu().tolist())
            self.GT_class.extend(GT_class.cpu().tolist())
            self.R_mask.extend(R.detach().cpu().numpy())
            self.GT_mask.extend(GT[1].detach().cpu().numpy())
            self.length += 1
            
            return [self.loss, self.Acc, self.RC, self.PC, self.OIS, self.IoU, self.AIU, self.mIoU, self.R_class, self.R_mask, self.DC, self.RC_all, self.PC_all, self.GT_class, self.GT_mask, self.length]
            
        else:
            self.MAE += (1 - torch.mean(torch.abs(R - GT[1]))).cpu().numpy()
            self.MSE += (1 - torch.mean((R - GT[1])**2)).cpu().numpy()
            self.length += 1
            
            return [self.loss, self.MAE, self.MSE, self.length]
            
            
    def metrics_avg(self, metric, training_mode):
        loss = (metric[0]/metric[-1])
        if training_mode != 'FSS i':
            Acc = (metric[1]/metric[-1])*100
            RC = (metric[2]/metric[-1])*100
            PC = (metric[3]/metric[-1])*100
            F1 = ((2*RC*PC)/(RC+PC+1e-12))
            OIS = (metric[4]/metric[-1])*100
            IoU = (metric[5]/metric[-1])*100
            AIU = (metric[6]/metric[-1])*100
            mIoU = (metric[7]/metric[-1])*100
            Pe, Re, _ = precision_recall_curve(np.array(metric[14]).reshape(-1), np.array(metric[9]).reshape(-1))
            PRC = auc(Re, Pe)*100
            ROC_I = roc_auc_score(np.array(metric[13]), np.array(metric[8]))*100
            ROC_P = roc_auc_score(np.array(metric[14]).reshape(-1).astype(int), np.array(metric[9]).reshape(-1))*100
            DC = (metric[10]/metric[-1])*100
            
            return [loss, Acc, RC, PC, F1, OIS, IoU, AIU, mIoU, PRC, ROC_I, ROC_P, DC]
        
        else:
            MAE = (metric[1]/metric[-1])*100
            MSE = (metric[2]/metric[-1])*100
            
            return [loss, MAE, MSE]
            
