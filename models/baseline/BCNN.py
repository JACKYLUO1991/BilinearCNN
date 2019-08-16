#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/13/2019 11:40 AM
# @Author  : Aries
# @Site    :
# @File    : BCNN.py
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), Your Company
# @Contact : XXXXXX@gmail.com
# @Site    :

import torch
from torch import nn as nn
from torchvision.models import resnet50, vgg16
from torch.nn import functional as F


# class BilinearCNN(nn.Module):
#     # Bilinear CNN Models for Fine-grained Visual Recognition
#     def __init__(self):
#         super(BilinearCNN, self).__init__()
#         self.features = nn.Sequential(
#             resnet50().conv1,
#             resnet50().bn1,
#             resnet50().relu,
#             resnet50().maxpool,
#             resnet50().layer1,
#             resnet50().layer2,
#             resnet50().layer3,
#             resnet50().layer4
#         )
#         self.classifiers = nn.Linear(2048 ** 2, 200)
#
#     def forward(self, x):
#         x = self.features(x)
#         batch_size = x.size(0)
#         x = x.view(batch_size, 2048, x.size(2) ** 2)
#         # torch.bmm: 对存储在batch和batch内的矩阵进行批矩阵乘操作
#         x = (torch.bmm(x, torch.transpose(x, 1, 2)) / 28 ** 2).view(batch_size, -1)
#         # x = (torch.bmm(x, torch.transpose(x, 1, 2))).view(batch_size, -1)
#         x = F.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-5))
#         x = self.classifiers(x)
#         return x


class BilinearCNN(nn.Module):
    # https://github.com/HaoMood/bilinear-cnn/blob/master/src/bilinear_cnn_all.py
    def __init__(self):
        super(BilinearCNN, self).__init__()
        self.features = vgg16(pretrained=True).features
        self.features = nn.Sequential(
            *list(self.features.children())[:-1]  # Remove pool5
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 ** 2, 200)
        )
    

    def forward(self, x):
        x = self.features(x)
        batch_size = x.size(0)
        x = x.view(batch_size, x.size(1), x.size(2) ** 2)
        # torch.bmm: 对存储在batch和batch内的矩阵进行批矩阵乘操作
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / x.size(2) ** 2).view(batch_size, -1)
        x = F.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        x = self.fc(x)
        return x