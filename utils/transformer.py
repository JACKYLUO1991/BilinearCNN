#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/13/2019 7:23 PM
# @Author  : Aries
# @Site    : 
# @File    : transformer.py
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), Your Company
# @Contact : XXXXXX@gmail.com
# @Site    :

import torchvision


train_transforms = torchvision.transforms.Compose([
	torchvision.transforms.Resize((512, 512)),
	torchvision.transforms.RandomHorizontalFlip(),
	torchvision.transforms.RandomCrop((448, 448)),
	torchvision.transforms.ToTensor(),
	torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
	                                 std=(0.229, 0.224, 0.225))
])
test_transforms = torchvision.transforms.Compose([
	torchvision.transforms.Resize((512, 512)),
	torchvision.transforms.CenterCrop((448, 448)),
	torchvision.transforms.ToTensor(),
	torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
	                                 std=(0.229, 0.224, 0.225))
])
