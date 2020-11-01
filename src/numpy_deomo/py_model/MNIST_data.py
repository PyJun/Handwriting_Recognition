'''
用 python 定义一个 Mnist 的类，包含获 mnist 的所有数据
'''


import os
import numpy as np


class Mnist:
	def __init__(self,path):
		self.__load_mnist(path,kind='train')
		self.__load_mnist(path,kind='test')

	@property
	def train_images(self):
		return self.__train_images
	@property
	def train_labels(self):
		return self.__train_labels
	@property
	def test_images(self):
		return self.__test_images
	@property
	def test_labels(self):
		return self.__test_labels

	def __load_mnist(self,path,kind):
		labels_path=os.path.join(path,kind+'-labels.idx1-ubyte')
		images_path=os.path.join(path,kind+'-images.idx3-ubyte')
		with open(labels_path,'rb') as f1:
			f1.read(8)
			labels=np.fromfile(f1,dtype='uint8')
		with open(images_path,'rb') as f2:
			f2.read(16)
			images=np.fromfile(f2,dtype='uint8').reshape(len(labels),784)
		if kind=='train':
			self.__train_images=images/255
			self.__train_labels=self.__one_hot(labels)
		elif kind=='test':
			self.__test_images=images/255
			self.__test_labels=self.__one_hot(labels)

	def __one_hot(self,labels):
		onehot=np.zeros([labels.shape[0],10])
		for idx,val in enumerate(labels):
			onehot[idx,val]=1
		return onehot 
