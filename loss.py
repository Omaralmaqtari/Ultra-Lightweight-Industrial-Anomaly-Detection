import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt


class DiceLoss(nn.Module):
    def __init__(self, threshold):
        super(DiceLoss, self).__init__()
        self.threshold = threshold

    def forward(self, SR, GT, smooth=1e-12): 
        SR = nn.Sigmoid()(SR)
        SR = SR.view(-1)
        GT = GT.reshape(-1)
        
        Inter = torch.sum((SR>self.threshold)&(GT>self.threshold))
        Union = torch.sum(SR>self.threshold) + torch.sum(GT>self.threshold)
        Dice = float(2.*Inter)/(float(Union) + smooth)
        
        return 1 - Dice

class IoULoss(nn.Module):
    def __init__(self, threshold):
        super(IoULoss, self).__init__()
        self.threshold = threshold

    def forward(self, SR, GT, smooth=1e-12):
        SR = nn.Sigmoid()(SR)
        SR = SR.view(-1)
        GT = GT.reshape(-1)
        
        Inter = torch.sum((SR>self.threshold)&(GT>self.threshold))
        Union = torch.sum(SR>self.threshold) + torch.sum(GT>self.threshold) - Inter
        IoU = float(Inter)/(float(Union) + smooth)
                
        return 1 - IoU

class mIoULoss(nn.Module):
    def __init__(self, threshold):
        super(mIoULoss, self).__init__()
        self.threshold = threshold
    
    def forward(self, SR, GT, smooth=1e-12):
        SR = nn.Sigmoid()(SR)
        SR = SR.view(-1)
        GT = GT.reshape(-1)
        
        # IoU of Foreground
        Inter1 = torch.sum((SR>self.threshold)&(GT>self.threshold))
        Union1 = torch.sum(SR>self.threshold) + torch.sum(GT>self.threshold) - Inter1
        IoU1 = float(Inter1)/(float(Union1) + smooth)
        
        # IoU of Background
        Inter2 = torch.sum((SR<self.threshold)&(GT<self.threshold))
        Union2 = torch.sum(SR<self.threshold) + torch.sum(GT<self.threshold) - Inter2
        IoU2 = float(Inter2)/(float(Union2) + smooth)

        mIoU = (IoU1 + IoU2) / 2
                
        return 1 - mIoU

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=4, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, SR, GT):
        BCE_loss = self.criterion(SR, GT)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()


def PRC(RC, PC, result_path, report_name):
    RC1 = []
    PC1 = []
    RC = list(map(list, zip(*RC)))
    PC = list(map(list, zip(*PC)))
    
    for i in range(len(RC)):
        RC1.append(np.sum(RC[i])/len(RC[i]))
    for i in range(len(PC)):
        PC1.append(np.sum(PC[i])/len(PC[i]))
        
    RC = np.fliplr([RC1])[0]  #to avoid getting negative AUC
    PC = np.fliplr([PC1])[0]  #to avoid getting negative AUC
    AUPRC = np.trapz(PC, RC)
    print("\nAUPRC: " +str(AUPRC))
    plt.figure()
    plt.plot(RC,PC,'-',label='AUPRC = %0.4f' % AUPRC)
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(result_path+report_name+'_PRC.png')
    
    return RC, PC

    
def displayfigures(results, result_path, report_name):
    for i in range(len(results)):
        plt.Figure()
        plt.plot(results[i][1], marker='o', markersize=3, label="Train "+results[i][0])
        plt.plot(results[i][2], marker='o', markersize=3, label="Valid "+results[i][0])
        plt.legend(loc="lower right")
        plt.xlabel("Epochs")
        plt.ylabel(results[i][0]+"%")
        if results[i][0] != "Loss":
            plt.ylim(0,100)
        plt.savefig(result_path+report_name+'_'+results[i][0]+'_results.png')
        plt.show()