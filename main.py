#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/14/2019 10:46 AM
# @Author  : Aries
# @Site    : 
# @File    : main.py
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), Your Company
# @Contact : XXXXXX@gmail.com
# @Site    :
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR, MultiStepLR
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import os
import os.path as osp
import shutil
from utils import *
from models.baseline.BCNN import BilinearCNN
from datasets.cub2011_dataset import CUB2011DataSet

from train import train
from validate import validate
from torchvision import models

import argparse

parser = argparse.ArgumentParser(description='PyTorch Fine-grained Training')
parser.add_argument('--root_dir', type=str, default='./datasets/CUB2011',
                    help='Root of the data set')
parser.add_argument('--init_type', default='kaiming', type=str,
                    metavar='INIT', help='init net')
parser.add_argument('--lr', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='momentum', help='momentum')
parser.add_argument('--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=180, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: None)')
parser.add_argument('--checkpoints', default='checkpoints', type=str,
                    help='director of checkpoint')
args = parser.parse_args()


def save_checkpoint(state, is_best, filename):
	if isinstance(filename, list) and len(filename) == 2:
		torch.save(state, filename[0])
		if is_best:
			shutil.copyfile(filename[0], filename[1])
	else:
		raise Exception("save deploy error, please check files...")


def main():
	# 迭代记录最好的模型参数
	best_prec1 = 0
	
	train_dataset = CUB2011DataSet(base_path=args.root_dir, is_train=True, transform=train_transforms)
	validate_dataset = CUB2011DataSet(base_path=args.root_dir, transform=test_transforms)
	
	model = BilinearCNN()
	# init_net(model, init_type=args.init_type)
	# model = models.resnet50(pretrained=True)
	# model.fc = nn.Linear(in_features=model.fc.in_features, out_features=200)
	
	criterion = nn.CrossEntropyLoss()
	# 更新需要训练的模型参数
	# https://www.aiuai.cn/aifarm615.html
	ignored_params = list(map(id, model.fc.parameters()))
	base_params = filter(lambda p: id(p) not in ignored_params,
	                     model.parameters())
	
	optimizer = torch.optim.SGD([
		{'params': base_params},
		{'params': model.fc.parameters(), 'lr': args.lr}
	], lr=args.lr * 10, momentum=args.momentum)
	
	cudnn.benchmark = True
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	criterion = criterion.to(device)
	
	if args.resume:
		if os.path.isfile(args.resume):
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print('Load Network  <==> Continue from {} epoch {}'.format(args.resume, checkpoint['epoch']))
		else:
			print('Load Network  <==> Failed')
	
	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
	                              pin_memory=True)
	val_dataloader = DataLoader(validate_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
	print("Loading model already...")
	scheduler = MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)
	
	if not osp.exists(args.checkpoints):
		os.mkdir(args.checkpoints)
	
	writer = SummaryWriter(log_dir=args.checkpoints)
	print("Start training...")
	for epoch in range(args.start_epoch, args.epochs):
		scheduler.step(epoch)
		
		loss, top1, top5 = train(train_dataloader, model, criterion, optimizer, epoch, device)
		test_loss, prec1, prec5 = validate(val_dataloader, model, criterion, epoch, device)
		
		# TODO：训练曲线绘制
		writer.add_scalar('Loss/train', loss, epoch + 1)
		writer.add_scalar('Loss/test', test_loss, epoch + 1)
		
		writer.add_scalar('Acc/train/top1', top1, epoch + 1)
		writer.add_scalar('Acc/train/top5', top5, epoch + 1)
		writer.add_scalar('Acc/test/top1', prec1, epoch + 1)
		writer.add_scalar('Acc/test/top5', prec5, epoch + 1)
		
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		
		filename = []
		filename.append(os.path.join(args.checkpoints, 'net-epoch-%s.pth.tar' % (epoch + 1)))
		filename.append(os.path.join(args.checkpoints, 'model_best.pth.tar'))
		
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_prec1': best_prec1,
			'optimizer': optimizer.state_dict(),
		}, is_best, filename)
	
	print("Training finished...")
	print(f"best val acc: {best_prec1:.3f}")


if __name__ == '__main__':
	main()
