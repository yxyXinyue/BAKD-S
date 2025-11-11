# --coding:utf-8--
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchsummary
from utils.config import config
import torch
import torchvision.models as models
import torchsummary

from tensorboardX import SummaryWriter


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=config.num_classes, deep_base=False, stem_width=32):
        self.inplanes = stem_width * 2 if deep_base else 64

        super(ResNet, self).__init__()
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_ = nn.Linear(512 * block.expansion, num_classes)
        #self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):  # block是BasicBlock或Bottleneck
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1_1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_(x)
        x = self.softmax(x)

        return x


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2,2,2,2], **kwargs)
    model_dict = model.state_dict()

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}#将 model_dict 里不属于 model_state_dict 的键剔除掉
        model_dict.update(pretrained_dict)#用预训练模型的参数字典 对 新模型的参数字典 model_state_dict 进行更新
        model.load_state_dict(model_dict)#将更新了参数的字典 “放”回到网络中
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='/home/yuanyuan/ROP/DLProjects/classifacation/pretrained'))
        print('===> Pretrain Model Have Been Loaded, Please fasten your seat belt and get ready to take off!')
    return model

def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    model_dict = model.state_dict()

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'],
                                             model_dir='/home/yy/ROP/DLProjects/classifacation/pretrained')
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}#将 model_dict 里不属于 model_state_dict 的键剔除掉
        model_dict.update(pretrained_dict)#用预训练模型的参数字典 对 新模型的参数字典 model_state_dict 进行更新
        model.load_state_dict(model_dict)#将更新了参数的字典 “放”回到网络中
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='/home/yuanyuan/ROP/DLProjects/classifacation/pretrained'))
        print('===> Pretrain Model Have Been Loaded, Please fasten your seat belt and get ready to take off!')
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    model_dict = model.state_dict()

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}#将 model_dict 里不属于 model_state_dict 的键剔除掉
        model_dict.update(pretrained_dict)#用预训练模型的参数字典 对 新模型的参数字典 model_state_dict 进行更新
        model.load_state_dict(model_dict)#将更新了参数的字典 “放”回到网络中
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='/home/yuanyuan/ROP/DLProjects/classifacation/pretrained'))
        print('===> Pretrain Model Have Been Loaded, Please fasten your seat belt and get ready to take off!')
    return model

def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']),strict=False)
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']),strict=False)
    return model

def resnet118(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2,2,2,2], **kwargs)
    model_dict = model.state_dict()

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'],
                                             model_dir='/home/yy/ROP/DLProjects/classifacation/pretrained')
        keys = []
        for k, v in pretrained_dict.items():
               keys.append(k)
        print(len(keys))
        print(keys[95:])
        i = 0
         
        # 自己网络和预训练网络结构一致的层，使用预训练网络对应层的参数初始化
        for k, v in model_dict.items():
            if v.size() == pretrained_dict[keys[i]].size():
                 model_dict[k] = pretrained_dict[keys[i]]
                 #print(model_dict[k])
                 i = i + 1
        print(i)
        print(keys[i])
        #model.load_state_dict(model_dict)
        model.load_state_dict(model_dict,strict=False)#将更新了参数的字典 “放”回到网络中
        #model.load_state_dict(model_zoo.load_url(model_urls['densenet121']))
        print('===> Pretrain Model Have Been Loaded, Please fasten your seat belt and get ready to take off!')
    return model




