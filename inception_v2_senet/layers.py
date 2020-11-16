import os
import time
import copy
from collections import defaultdict
from tqdm import tqdm
import shutil
import zipfile
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils

import cv2
from PIL import Image
# from skimage import io
# from skimage import transform
#from skimage import io, transform
#import albumentations
#from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
#from albumentations.pytorch import ToTensor



class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SE_block(nn.Module):
    def __init__(self, in_channels, ratio):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//ratio, bias=True),
            nn.BatchNorm1d(in_channels//ratio, eps=1e-5),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels//ratio, in_channels, bias=True),
            nn.BatchNorm1d(in_channels, eps=1e-5),
            nn.Sigmoid()            
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
#         x = x * w.expand_as(x)
#         return x
        return w * x

class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.conv_2 = Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        self.conv_3 = Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.mixed_4a = nn.MaxPool2d(3, stride=2, padding=1)
        self.mixed_4b = Conv2d(64, 96, 3, stride=2, padding=1, bias=False)
        self.mixed_5a = nn.Sequential(
            Conv2d(160, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 96, 3, stride=1, padding=1, bias=False)
        )
        self.mixed_5b = nn.Sequential(
            Conv2d(160, 64, 1, stride=1, padding=0, bias=False),
            Conv2d(64, 64, (1,7), stride=1, padding=(0,3), bias=False),
            Conv2d(64, 64, (7,1), stride=1, padding=(3,0), bias=False),
            Conv2d(64, 96, 3, stride=1, padding=1, bias=False)
        )        
        self.mixed_6a = Conv2d(192, 192, 3, stride=2, padding=1, bias=False)
        self.mixed_6b = nn.MaxPool2d(3, stride=2, padding=1)
        
    def forward(self, x):
        #256x256x3
        x = self.conv_1(x)#128x128x32
        x = self.conv_2(x)#128x128x32
        x = self.conv_3(x)#128x128x64
        x = torch.cat((self.mixed_4a(x), self.mixed_4b(x)), dim=1)#64x64x160
        x = torch.cat((self.mixed_5a(x), self.mixed_5b(x)), dim=1)#64x64x192
        x = torch.cat((self.mixed_6a(x), self.mixed_6b(x)), dim=1)#32x32x384
        return x
    
class Inception_A_v2_SE(nn.Module):
    
    def __init__(self, scale=1):
        super().__init__()
        self.mixed_1a = Conv2d(384, 32, 1, stride=1, padding=0, bias=False)
        self.mixed_1b = nn.Sequential(
            Conv2d(384, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        )
        self.mixed_1c = nn.Sequential(
            Conv2d(384, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 48, 3, stride=1, padding=1, bias=False),
            Conv2d(48, 64, 3, stride=1, padding=1, bias=False)
        )
        self.conv_2 = Conv2d(128, 384, 1, stride=1, padding=0, bias=True)
        self.SE_3 = SE_block(384, 16)
        self.relu = nn.ReLU(inplace=False)
        self.scale = nn.parameter.Parameter(torch.tensor(float(scale)))
        
    def forward(self, x):
        x_res = torch.cat((self.mixed_1a(x), self.mixed_1b(x), self.mixed_1c(x)),dim=1)
        x_res = self.conv_2(x_res)
        x_res = self.SE_3(x_res)
        x = self.relu(x + x_res * self.scale)
        return x  #32x32x384
        
class Reduction_A_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixed_1a = nn.MaxPool2d(3, stride=2, padding=1)
        self.mixed_1b = Conv2d(384, 384, 3, stride=2, padding=1, bias=False)
        self.mixed_1c = nn.Sequential(
            Conv2d(384, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            Conv2d(256, 384, 3, stride=2, padding=1, bias=False)
        )
        
    def forward(self, x):
        x = torch.cat((self.mixed_1a(x), self.mixed_1b(x), self.mixed_1c(x)),dim=1)#16x16x1152
        return x
        
class Inception_B_v2_SE(nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.mixed_1a = Conv2d(1152, 192, 1, stride=1, padding=0, bias=False)
        self.mixed_1b = nn.Sequential(
            Conv2d(1152, 128, 1, stride=1, padding=0, bias=False),
            Conv2d(128, 160, (1,7), stride=1, padding=(0,3), bias=False),
            Conv2d(160, 192, (7,1), stride=1, padding=(3,0), bias=False)
        )
        self.conv_2 = Conv2d(384, 1152, 1, stride=1, padding=0, bias=True)
        self.SE_3 = SE_block(1152, 16)
        self.relu = nn.ReLU(inplace=False)
        self.scale = nn.parameter.Parameter(torch.tensor(float(scale)))
        
    def forward(self, x):
        x_res = torch.cat((self.mixed_1a(x), self.mixed_1b(x)), dim=1)
        x_res = self.conv_2(x_res)
        x_res = self.SE_3(x_res)
        x = self.relu(x + x_res * self.scale)#16x16x1152
        return x

class Reduction_B_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixed_1a = nn.MaxPool2d(3, stride=2, padding=1)
        self.mixed_1b = nn.Sequential(
            Conv2d(1152, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 384, 3, stride=2, padding=1, bias=False)
        )
        self.mixed_1c = nn.Sequential(
            Conv2d(1152, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 288, 3, stride=2, padding=1, bias=False)
        )
        self.mixed_1d = nn.Sequential(
            Conv2d(1152, 256, 1, stride=1, padding=0, bias=False),
            Conv2d(256, 288, 3, stride=1, padding=1, bias=False),
            Conv2d(288, 320, 3, stride=2, padding=1, bias=False)
        )
        
    def forward(self, x):
        x = torch.cat((self.mixed_1a(x), self.mixed_1b(x), self.mixed_1c(x), self.mixed_1d(x)), dim=1)
        #8x8x2144
        return x
        
class Inception_C_v2_SE(nn.Module):
    def __init__(self, scale=1, activation=True):
        super().__init__()
        self.mixed_1a = Conv2d(2144, 192, 1, stride=1, padding=0, bias=False)
        self.mixed_1b = nn.Sequential(
            Conv2d(2144, 192, 1, stride=1, padding=0, bias=False),
            Conv2d(192, 224, (1,3), stride=1, padding=(0,1), bias=False),
            Conv2d(224, 256, (3,1), stride=1, padding=(1,0), bias=False)
        )
        self.conv_2 = Conv2d(448, 2144, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=False)
        self.SE_3 = SE_block(2144, 16)
        self.scale = nn.parameter.Parameter(torch.tensor(float(scale)))
        self.activation = activation
        
    def forward(self, x):
        x_res = torch.cat((self.mixed_1a(x), self.mixed_1b(x)), dim=1)
        x_res = self.conv_2(x_res)
        x_res = self.SE_3(x_res)
        x = x + x_res * self.scale
        #8x8x2144
        if self.activation:
            return self.relu(x)
        return x
        
class Inception_ResNet_v2_SE(nn.Module):
    def __init__(self, scale=[1,1,1]):
        super().__init__()
        blocks = []
        blocks.append(Stem())
        for i in range(5):
            blocks.append(Inception_A_v2_SE(scale[0]))
#         blocks.append(Inception_A_v2_SE(scale[0]))
        blocks.append(Reduction_A_v2())
        for i in range(10):
            blocks.append(Inception_B_v2_SE(scale[1]))
#         blocks.append(Inception_B_v2_SE(scale[1]))
        blocks.append(Reduction_B_v2())
        for i in range(4):
            blocks.append(Inception_C_v2_SE(scale[2], True))
#         blocks.append(Inception_C_v2_SE(scale[2], True))
        blocks.append(Inception_C_v2_SE(scale[2], False))
        self.features = nn.Sequential(*blocks)
        self.global_average_polling = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2144, 512),
            nn.BatchNorm1d(512, eps=1e-5),
            nn.ReLU(inplace=False),
            nn.Dropout(0.8, inplace=False)
        )
        self.fc2 = nn.Linear(512, 100)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.features(x)#8x8x2144
        x = self.global_average_polling(x)#1x1x2144
        x = self.fc1(x)#512
        x = self.fc2(x)#100
        x = self.output(x)
        return x

