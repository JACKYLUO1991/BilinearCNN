#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/12/2019 7:33 PM
# @Author  : JackyLUO
# @Site    : 
# @File    : train.py
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), Your Company
# @Contact : XXXXXX@gmail.com
# @Site    :
from __future__ import print_function, division

import time
from utils import *
from progress.bar import Bar


def train(train_loader, model, criterion, optimizer, epoch, device):
	# 模型进入训练模式
	model.train()
	
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	end = time.time()
	
	bar = Bar('Processing train', max=len(train_loader))
	
	for i, (data, target) in enumerate(train_loader):
		# 计算加载数据的时间
		data_time.update(time.time() - end)
		
		# 备注: 分布式训练中要设置non_blocking=True
		data = data.to(device)
		target = target.to(device)
		
		output = model(data)
		loss = criterion(output, target)
		
		prec1, prec5 = accuracy(output, target, topk=(1, 5))
		losses.update(loss.item(), data.size(0))
		top1.update(prec1[0], data.size(0))
		top5.update(prec5[0], data.size(0))
		
		# 计算一个批量计算的时间
		batch_time.update(time.time() - end)
		end = time.time()
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		bar.suffix = '(Epoch:{epoch} - {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.2f} | Acc@1: {acc1:.2f} | Acc@5: {acc5:.2f}'.format(
			batch=i + 1,
			size=len(train_loader),
			data=data_time.avg,
			bt=batch_time.avg,
			loss=losses.avg,
			acc1=top1.avg,
			acc5=top5.avg,
			epoch=epoch
		)
		bar.next()
	bar.finish()
	
	return losses.avg, top1.avg, top5.avg
