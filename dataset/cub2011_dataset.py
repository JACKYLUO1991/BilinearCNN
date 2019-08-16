#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/13/2019 5:08 PM
# @Author  : Aries
# @Site    : 
# @File    : cub2011_dataset.py
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), Your Company
# @Contact : XXXXXX@gmail.com
# @Site    :

from torch.utils.data import Dataset
import os.path as osp

from PIL import Image


class CUB2011DataSet(Dataset):
	"""
	Read cub-2011 dataset
	"""
	
	def __init__(self, base_path, transform=None, is_train=False):
		images_info = []
		
		if is_train:
			with open(osp.join(base_path, "lists/train.txt")) as f:
				for line in f:
					line = line.strip('\n')
					line = line.rstrip()
					words = line.split('.')
					images_info.append((line, int(words[0]) - 1))
		else:
			with open(osp.join(base_path, "lists/test.txt")) as f:
				for line in f:
					line = line.strip('\n')
					line = line.rstrip()
					words = line.split('.')
					images_info.append((line, int(words[0]) - 1))
		
		self.images_info = images_info
		self.transform = transform
		self.base_path = base_path
	
	def __getitem__(self, index):
		image_path, label = self.images_info[index]
		im = Image.open(osp.join(self.base_path, "images", image_path))
		im = im.convert('RGB')  # 有些图像可能是单通道的
		if self.transform is not None:
			im = self.transform(im)
		return im, label


	def __len__(self):
		return len(self.images_info)
