#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/12/2019 7:33 PM
# @Author  : JackyLUO
# @Site    : 
# @File    : validate.py
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), Your Company
# @Contact : XXXXXX@gmail.com
# @Site    :
from __future__ import print_function, division

import torch

import time
from utils import *
from progress.bar import Bar


def validate(val_loader, model, criterion, epoch, device):
	model.eval()
	
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	end = time.time()
	
	bar = Bar('Processing val', max=len(val_loader))
	
	with torch.no_grad():
		for i, (data, target) in enumerate(val_loader):
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
			
			bar.suffix = '(Epoch:{epoch} - {batch}/{size}) Batch: {bt:.3f}s | Loss: {loss:.3f} | Acc@1: {acc1:.2f} | Acc@5: {acc5:.2f}'.format(
				batch=i + 1,
				size=len(val_loader),
				bt=batch_time.avg,
				loss=losses.avg,
				acc1=top1.avg,
				acc5=top5.avg,
				epoch=epoch
			)
			bar.next()
		bar.finish()
		
		print('* Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
		      .format(top1=top1, top5=top5))
		
		return losses.avg, top1.avg, top5.avg
