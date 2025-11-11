# densenet原文地址 https://arxiv.org/abs/1608.06993
# densenet介绍 https://blog.csdn.net/zchang81/article/details/76155291
# 以下代码就是densenet在torchvision.models里的源码，为了提高自身的模型构建能力尝试分析下源代码：
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
}  # 这个是预训练模型可以在下边的densenet121,169等里直接在pretrained=True就可以下载


class _DenseLayer(nn.Sequential):  # 这是denselayer，也是nn.Seqquential，看来要好好学习用法 #
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),  # 这里要看到denselayer里其实主要包括两个卷积层，而且他们的channel数值得关注 #
        self.add_module('relu1', nn.ReLU(inplace=True)),  # 其实在add_module后边的逗号可以去掉，没有任何意义，又不是构成元组徒增歧义 #
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),  # 这里注意的是输出的channel数是growth_rate #
        self.drop_rate = drop_rate

    def forward(self, x):  # 这里是前传，主要解决的就是要把输出整形，把layer的输出和输入要cat在一起 #
        new_features = super(_DenseLayer, self).forward(x)  # #
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)  # 加入dropout增加泛化 #
        return torch.cat([x, new_features], 1)  # 在channel上cat在一起，以形成dense连接 #


class _DenseBlock(nn.Sequential):  # 是nn.Sequential的子类，将一个block里的layer组合起来 #
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size,
                                drop_rate)  # 后一层的输入channel是该denseblock的输入channel数，加上该层前面层的channnel数的和 #
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):  # 是nn.Sequential的子类，#这个就比较容易了，也是以后自己搭建module的案例#
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):  # 这就是densenet的主类了，看继承了nn.Modele类 #
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper) #每个denseblock里应该，每个Layer的输出特征数，就是论文里的k #
        block_config (list of 4 ints) - how many layers in each pooling block  #每个denseblock里layer层数, block_config的长度表示block的个数 #
        num_init_features (int) - the number of filters to learn in the first convolution layer   #初始化层里卷积输出的channel数#
        bn_size (int) - multiplicative factor for number of bottle neck layers   #这个是在block里一个denselayer里两个卷积层间的channel数 需要bn_size*growth_rate  #
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer  #dropout的概率，正则化的方法 #
        num_classes (int) - number of classification classes   #输出的类别数，看后边接的是linear，应该最后加损失函数的时候应该加softmax，或者交叉熵，而且是要带计算概率的函数 #
    """

    # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),)

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=config.num_classes):

        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.block_config = block_config
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Each denseblock 创建densebloc
        self.layer1 = self._make_layer(_DenseBlock, self.block_config[0], 64)
        self.layer2 = self._make_layer(_DenseBlock, self.block_config[1], 128)
        self.layer3 = self._make_layer(_DenseBlock, self.block_config[2], 256)
        self.layer4 = self._make_layer(_DenseBlock, self.block_config[3], 512)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Linear layer
        self.classifier = nn.Linear(1024, config.num_classes)
        #        self.classifier1 = nn.Linear(1024, 1)
        #        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, num_layers, num_features):  # block是BasicBlock或Bottleneck
        layers = []

        layers.append(block(num_layers=num_layers, num_input_features=num_features,
                            bn_size=self.bn_size, growth_rate=self.growth_rate, drop_rate=self.drop_rate))
        num_input_features = num_features + num_layers * self.growth_rate
        if num_layers != self.block_config[3]:
            layers.append(
                _Transition(num_input_features=num_input_features, num_output_features=num_input_features // 2))
        else:
            layers.append(nn.BatchNorm2d(1024))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out1 = self.relu(x4)
        out1 = self.avgpool(out1)
        out1 = out1.view(out1.size(0), -1)
        out = self.classifier(out1)
        out = self.softmax(out)

        return out

    def load_pretrained_weights(self):
        pretrained_dict = model_zoo.load_url(model_urls['densenet121'])
        model_dict = self.state_dict()
        keys = []
        for k, v in pretrained_dict.items():
            keys.append(k)
        i = 0
        # 自己网络和预训练网络结构一致的层，使用预训练网络对应层的参数初始化
        for k, v in model_dict.items():
            if v.size() == pretrained_dict[keys[i]].size():
                model_dict[k] = pretrained_dict[keys[i]]
                i = i + 1
        self.load_state_dict(model_dict, strict=False)


def densenet121(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet()
    if pretrained:
        model.load_pretrained_weights()
        print('===> Pretrain Model Have Been Loaded, Please fasten your seat belt and get ready to take off!')
    return model


def densenet161(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(growth_rate=32, block_config=(4, 6, 12, 8))
    if pretrained:
        # 这里简单使用 densenet121 的预训练权重，实际中可能需要修改为 densenet161 的加载逻辑
        model.load_pretrained_weights()
        print('===> Pretrain Model Have Been Loaded, Please fasten your seat belt and get ready to take off!')
    return model
