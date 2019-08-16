#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/13/2019 5:45 PM
# @Author  : Aries
# @Site    : 
# @File    : train_test_split.py
# @Software: PyCharm
# @Desc     :
# @license : Copyright(C), Your Company
# @Contact : XXXXXX@gmail.com
# @Site    :


def main():
	train_id = []
	test_id = []
	# image_id_label = {}
	train_path_label = []
	test_path_label = []
	
	# 将数据集分为训练和测试集
	with open("CUB2011/extra/train_test_split.txt", "r") as f:
		for line in f:
			image_id, label = line.strip("\n").split()
			if label == '1':
				train_id.append(image_id)
			elif label == '0':
				test_id.append(image_id)
			else:
				print("error label id: {}".format(image_id))
	
	# # 数据集ID和标签对应
	# with open("CUB2011/extra/image_class_labels.txt", "r") as fn:
	# 	for line in fn:
	# 		image_id, class_id = line.strip("\n").split()
	# 		image_id_label[image_id] = class_id
			
	# 路径和标签对应
	with open("CUB2011/extra/images.txt", "r") as ft:
		for line in ft:
			image_id, image_name = line.strip('\n').split()
			# label = image_id_label[image_id]
			if image_id in train_id:
				train_path_label.append((image_name))
			else:
				test_path_label.append((image_name))
	
	# Training number: 5994, Testing number: 5794
	print("Training number: {}".format(len(train_path_label)))
	print("Testing number: {}".format(len(test_path_label)))
	
	# 写入对应的训练和测试txt文件中
	with open("CUB2011/lists/train.txt", "w") as ftw:
		for line in train_path_label:
			ftw.write(line + '\n')
	with open("CUB2011/lists/test.txt", "w") as ftw2:
		for line in test_path_label:
			ftw2.write(line + '\n')


if __name__ == '__main__':
	main()