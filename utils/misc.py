#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/14/2019 10:49 AM
# @Author  : Aries
# @Site    : 
# @File    : misc.py
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), Your Company
# @Contact : XXXXXX@gmail.com
# @Site    :
# https://github.com/songdejia/DFL-CNN/blob/master/utils/init.py
from torch.nn import init


def init_net(net, init_type='normal'):
	init_weights(net, init_type)
	return net


def init_weights(net, init_type='normal', gain=0.02):
	def init_func(m):
		# this will apply to each layer
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # good for relu
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, gain)
			init.constant_(m.bias.data, 0.0)
	net.apply(init_func)


class AverageMeter(object):
	"""Computes and stores the average and current value
	   Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
	"""
	
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
