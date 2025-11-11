#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 19:16:58 2020

@author: yy
"""
#densenet原文地址 https://arxiv.org/abs/1608.06993 
#densenet介绍 https://blog.csdn.net/zchang81/article/details/76155291
#以下代码就是densenet在torchvision.models里的源码，为了提高自身的模型构建能力尝试分析下源代码：
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from utils.config import config

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}   #这个是预训练模型可以在下边的densenet121,169等里直接在pretrained=True就可以下载 

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class _TransitionLayer(nn.Module):
    def __init__(self, inplace, plance):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace,out_channels=plance,kernel_size=1,stride=1,padding=0,bias=False),
            nn.AvgPool2d(kernel_size=2,stride=2),
        )

    def forward(self, x):
        return self.transition_layer(x)


class _DenseLayer(nn.Module):
    def __init__(self, inplace, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.cat([x, y], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, inplances, growth_rate, bn_size , drop_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(inplances + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DenseNet(nn.Module):
    def __init__(self, init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16],num_classes=config.num_classes):
        super(DenseNet, self).__init__()
        bn_size = 4
        drop_rate = 0
        self.conv1 = Conv1(in_planes=3, places=init_channels)

        blocks*4

        num_features = init_channels
        self.layer1 = DenseBlock(num_layers=blocks[0], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        self.transition1 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
        self.layer2 = DenseBlock(num_layers=blocks[1], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        self.transition2 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
        self.layer3 = DenseBlock(num_layers=blocks[2], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        self.transition3 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
        self.layer4 = DenseBlock(num_layers=blocks[3], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, num_classes)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.conv1(x)
        
        x1 = self.layer1(x)
        x1 = self.transition1(x1)
        x2 = self.layer2(x1)
        x2 = self.transition2(x2)
        x3 = self.layer3(x2)
        x3 = self.transition3(x3)
        x4 = self.layer4(x3)

        out1 =self.avgpool(x4)
        out1 = out1.view(out1.size(0),-1)
        out = self.fc(out1)
        out = self.softmax(out)
        
        return out

def densenet169(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 32, 32])
#    best_model = torch.load('/home/yy/ROP/DS_classification/checkpoints/best_model/pretrained/3_model_best.pth.tar')
#    model.load_state_dict(best_model["state_dict"])
    model_dict = model.state_dict()

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['densenet169'])
        keys = []
        for k, v in pretrained_dict.items():
               keys.append(k)
        i = 0
        # 自己网络和预训练网络结构一致的层，使用预训练网络对应层的参数初始化
        for k, v in model_dict.items():
            if v.size() == pretrained_dict[keys[i]].size():
                 model_dict[k] = pretrained_dict[keys[i]]
                 #print(model_dict[k])
                 i = i + 1
        model.load_state_dict(model_dict,strict=False)#将更新了参数的字典 “放”回到网络中
        #model.load_state_dict(model_zoo.load_url(model_urls['densenet121']))
        print('===> Pretrain Model Have Been Loaded, Please fasten your seat belt and get ready to take off!')
    return model