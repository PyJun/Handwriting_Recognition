'''
观察模型在训练集和测试集上的准确率
'''


import os
import pickle
import matplotlib.pyplot as plt
from mnist_train import load_model, NoModelError
from py_model.NN_model import *
from py_model.MNIST_data import *


def calculate_accuracy(model):
	mnist=Mnist('../data/MNIST_uncompressed')
	x_train=np.where(mnist.train_images>0.4,1,0)
	x_test=np.where(mnist.test_images>0.4,1,0)
	y_train_=np.argmax(mnist.train_labels,axis=1)
	y_test_=np.argmax(mnist.test_labels,axis=1)
	y_train_pred=model.predict(x_train)
	y_test_pred=model.predict(x_test)
	acc_train=np.mean(y_train_pred==y_train_)
	acc_test=np.mean(y_test_pred==y_test_)
	return acc_train, acc_test


def show_figure(cost,acc_train,acc_test):
	fig,ax=plt.subplots(1,2,figsize=(9,5))
	
	ax[0].bar(0.2,acc_train*100,0.2,facecolor='b')
	ax[0].bar(0.6,acc_test*100,0.2,facecolor='g')
	ax[0].set_ylim([90,101])
	ax[0].set_yticks(range(90,101))
	ax[0].set_xticks((0.2,0.6))
	ax[0].set_xticklabels(('训练集','测试集'),fontproperties='SimHei',fontsize=12)
	ax[0].set_xlabel('数据集类型',fontproperties='SimHei',fontsize=14)
	ax[0].set_ylabel('准确率（%）',fontproperties='SimHei',fontsize=14)
	ax[0].set_title('模型准确率柱形图',fontproperties='SimHei',fontsize=16)
	ax[0].text(0.2,acc_train*100+0.2,f'{acc_train:.2%}',ha='center',va='bottom')
	ax[0].text(0.6,acc_test*100+0.2,f'{acc_test:.2%}',ha='center',va='bottom')
	
	ax[1].plot(range(len(cost)),cost)
	ax[1].set_title('训练过程损失曲线',fontproperties='SimHei',fontsize=16)
	ax[1].set_ylabel('总损失',fontproperties='SimHei',fontsize=14)
	ax[1].set_xlabel('训练步数',fontproperties='SimHei',fontsize=14)
	plt.tight_layout()
	
	plt.show()


def main():
	try:
		nn=load_model('nn1.pkl')
	except NoModelError as err:
		print(err)
	else:
		acc_train,acc_test=calculate_accuracy(nn)
		show_figure(nn.cost_,acc_train,acc_test)


if __name__=='__main__':
	main()