'''
评估模型在自己数据集上的效果
'''


import os
import matplotlib.pyplot as plt
from mnist_train import load_model, NoModelError
from mnist_test import calculate_accuracy
from py_model.MY_handwriting import load_handwriting, NoDataError
from py_model.MNIST_data import *
from py_model.NN_model import *


def get_accuracy(model,handwriting):
	x,y_=handwriting['images'],handwriting['labels']
	x=np.where(x>0,1,0)
	y_pred=model.predict(x)
	acc_eval=np.mean(y_pred==y_)
	acc_train,acc_test=calculate_accuracy(model)
	acc=np.array([acc_train,acc_test,acc_eval])
	return acc


def show_accuracy(acc):
	xs=[0.1,0.5,0.9]
	for x,h,c in zip(xs,acc*100,('r','b','g')):
		plt.bar(x,h,0.2,facecolor=c)
		plt.text(x,h+0.2,f'{h/100:.2%}',ha='center',va='bottom',fontsize=12)
	plt.ylim([90,101])
	plt.yticks(range(90,101))
	plt.xticks(xs,('训练集','测试集','未知集'),fontproperties='SimHei',fontsize=12)
	plt.xlabel('数据集类型',fontproperties='SimHei',fontsize=15)
	plt.ylabel('准确率（%）',fontproperties='SimHei',fontsize=15)
	plt.title('模型准确率柱形图',fontproperties='SimHei',fontsize=16)
	plt.tight_layout()
	plt.show()


def show_prediction(model,handwriting,origin=True,shuffle=True):
	images,x,y_=handwriting['origin_images'],handwriting['images'],handwriting['labels']
	x=np.where(x>0,1,0)
	y_pred=model.predict(x)
	err_index=np.argwhere(y_pred!=y_)[:,0]
	for i in err_index:
			fig,ax=plt.subplots(1,2,figsize=(8,5))
			ax[0].imshow(images[i])
			ax[1].imshow(x[i].reshape(28,28),cmap='Greys',interpolation='nearest')
			for axi in ax:
				axi.set_xticks([])
				axi.set_yticks([])
			ax[0].set_title('手机拍摄图片',fontproperties='SimHei',fontsize=14)
			ax[1].set_title('归一化灰度图',fontproperties='SimHei',fontsize=14)
			fig.text(0.5,0.9,f'模型预测结果：{y_pred[i]}',fontproperties='SimHei',
						fontsize=20,ha='center',color='r')
			plt.tight_layout(pad=2)
			plt.show()
	if shuffle is True:
		idx=np.random.permutation(y_.shape[0])
		images,x,y_,y_pred=images[idx],x[idx],y_[idx],y_pred[idx]
	for i in range(2):
		fig,ax=plt.subplots(5,10,figsize=(10,5))
		ax=ax.flatten()
		for j in range(50):
			index=i*50+j
			if origin is True:
				ax[j].imshow(images[index])
			else:
				ax[j].imshow(x[index].reshape(28,28),cmap='Greys',
								interpolation='nearest')
			color='g' if y_pred[index]==y_[index] else 'r'
			ax[j].set_title(f'预测值:{y_pred[index]}',fontproperties='SimHei',
							fontsize=10,color=color)
		for axi in ax:
			axi.set_xticks([])
			axi.set_yticks([])
		plt.tight_layout()
		plt.show()


def main():
	try:
		nn=load_model('nn1.pkl')
		handwriting=load_handwriting('../data/my_handwriting')
	except (NoModelError,NoDataError) as err:
		print(err)
	else:
		acc=get_accuracy(nn,handwriting)
		show_accuracy(acc)
		show_prediction(nn,handwriting)


if __name__=='__main__':
	main()

	
