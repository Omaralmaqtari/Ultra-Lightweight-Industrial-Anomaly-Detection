
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import kornia.morphology as morph
from kornia.filters import laplacian

    
class FeatureExtraction(nn.Module):
    def __init__(self, ch_in, kernels=[3,5,7,9]):
        super(FeatureExtraction, self).__init__()
        
        self.ch_in = ch_in
        self.kernels = kernels
        
    def LoG(self, image, threshold):
        LoG1 = laplacian(image, (self.kernels[0],self.kernels[0]), border_type='constant')
        LoG2 = laplacian(image, (self.kernels[1],self.kernels[1]), border_type='constant')
        LoG3 = laplacian(image, (self.kernels[2],self.kernels[2]), border_type='constant')
        LoG4 = laplacian(image, (self.kernels[3],self.kernels[3]), border_type='constant')
        
        LoG = (LoG1 * LoG2) + LoG3 + LoG4
        LoG1 = []
        for b in range(LoG.shape[0]):
            LoG1.append(F.threshold(LoG[b].unsqueeze(0), (LoG[b].max()*threshold).item(), 0.))
        LoG = torch.cat(LoG1, 0)
        
        return LoG
    
    def Histogram(self, image, kernel, stride, ratio):
        threshold = kernel.shape[-1]*kernel.shape[-1]*ratio
        
        image[image>0.] = 1.
        
        segf = F.conv2d(image, kernel, stride=stride)
        segf = nn.Tanh()(F.threshold(segf, threshold, 0.))
        
        return segf
    
    def EF(self, image, threshold=[0.05,0.25], k_idx=0):
        img_shape = image.shape
        morph_kernel = torch.ones(self.kernels[k_idx], self.kernels[k_idx]).to(image.device)
        hk = torch.ones(self.ch_in,self.ch_in,self.kernels[k_idx],self.kernels[k_idx]).to(image.device)
        
        if self.ch_in == 1:
            image = T.Grayscale(num_output_channels=1)(image)
        
        image1 = []
        for b in range(image.shape[0]):
            image1.append(F.threshold(image[b].unsqueeze(0), (image[b].max()*threshold[0]).item(), 0.))
        image = torch.cat(image1, 0)
        
        LoG = self.LoG(image, threshold[0])
        
        segf = self.Histogram(LoG, hk, self.kernels[k_idx]-2, threshold[1])
        segf = morph.closing(segf, morph_kernel)
        
        if threshold[1] < 0.2:
            segf = morph.erosion(segf, morph_kernel)
        
        return F.interpolate(segf, size=img_shape[2:], mode='bilinear')
