#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 10:22:03 2020

@author: yy
"""
from torchvision import models
import torch.nn as nn
from utils.config import config
class VGGNet(nn.Module):
    def __init__(self, num_classes=config.num_classes):
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=True)
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
#net = models.vgg16(pretrained=False)
#print(net)