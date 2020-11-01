'''
训练神经网络，并画出其损失曲线
'''


import os
import pickle
import matplotlib.pyplot as plt
from py_model.NN_model import *
from py_model.MNIST_data import *


INPUT_MODE=784
OUTPUT_MODE=10
LAYER_MODE=500
L1=0
L2=0.1
LEARNING_STEPS=1500
LEARNING_RATE=0.001
LEARNING_RATE_DECAY=0.00001
N_BATCHES=60
ALPHA=0.001


class NoModelError(Exception):
	pass


def train(filename='nn1.pkl',replace=False):
	filepath=os.path.join('../data/model/py',filename)
	if os.path.exists(filepath) and replace is False:
		print(f'模型文件：{filepath} 已经存在！')
	else:
		mnist=Mnist('../data/MNIST_uncompressed')
		x=np.where(mnist.train_images>0.4,1,0)
		y_=mnist.train_labels
		nn=NeuralNet(
					INPUT_MODE, OUTPUT_MODE, LAYER_MODE, L1, L2,
					LEARNING_STEPS, LEARNING_RATE, N_BATCHES,
					LEARNING_RATE_DECAY, ALPHA
					)
		nn.train(x,y_)
		with open(filepath,'wb') as f:
			pickle.dump(nn,f)
		show_cost(nn.cost_)


def show_cost(cost):
	plt.figure(figsize=(8,5))
	plt.plot(range(len(cost)),cost)
	plt.ylabel('Cost')
	plt.xlabel('Step')
	plt.tight_layout()
	plt.show()


def load_model(filename='nn1.pkl'):
	filepath=os.path.join('../data/model/py',filename)
	if os.path.exists(filepath):
		with open(filepath,'rb') as f:
			nn=pickle.load(f)
		return nn
	else:
		raise NoModelError(f'需加载的模型文件： {filepath} 不存在！')


def main():
	train()
	input()


if __name__=='__main__':
	main()

