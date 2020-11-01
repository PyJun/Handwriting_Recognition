#查看验证集和测试集上训练模型的准确率


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from py_model.MY_handwriting import load_handwriting, NoDataError
from py_model.mnist_inference import *
from mnist_train import *


def get_onehot(y_):
	one_hot=np.zeros((y_.shape[0],10))
	for idx,var in enumerate(y_):
		one_hot[idx,var]=1
	return one_hot


def calculate_accuracy(mnist,handwriting,filename='NN.ckpt'):
	x=tf.placeholder(tf.float32,[None,INPUT_MODE],name='x-input')
	y_=tf.placeholder(tf.float32,[None,OUTPUT_MODE],name='y-input')
	y=inference(x,None)
	
	my_data_x=handwriting['images']
	my_data_y_=get_onehot(handwriting['labels'])
	
	train_feed={x:np.where(mnist.train.images>0.4,1,0),y_:mnist.train.labels} 
	validate_feed={x:np.where(mnist.validation.images>0.4,1,0),y_:mnist.validation.labels}
	test_feed={x:np.where(mnist.test.images>0.4,1,0),y_:mnist.test.labels}
	mydata_feed={x:np.where(my_data_x>0,1,0),y_:my_data_y_}
	
	correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	
	variables_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
	ema_variables=variables_averages.variables_to_restore()
	
	saver=tf.train.Saver(ema_variables)
	
	with tf.Session() as sess:
		if os.path.exists(os.path.join(FILEPATH,'checkpoint')):
			saver.restore(sess,os.path.join(FILEPATH,filename))
			train_acc=sess.run(accuracy,feed_dict=train_feed)
			validate_acc=sess.run(accuracy,feed_dict=validate_feed)
			test_acc=sess.run(accuracy,feed_dict=test_feed)
			mydata_acc=sess.run(accuracy,feed_dict=mydata_feed)
			acc=np.array([train_acc,validate_acc,test_acc,mydata_acc])
			y_pred_mydata=sess.run(tf.argmax(y,1),feed_dict=mydata_feed)
			return acc,y_pred_mydata
		else:
			raise NoModelError(f'模型文件：{os.path.join(FILEPATH,filename)} 不存在！')
	
	
def show_accuracy(acc):
	xs=[0.1,0.5,0.9,1.3]
	for x,h,c in zip(xs,acc*100,('r','b','g','y')):
		plt.bar(x,h,0.2,facecolor=c)
		plt.text(x,h+0.2,f'{h/100:.2%}',ha='center',va='bottom',fontsize=12)
	plt.ylim([90,101])
	plt.yticks(range(90,101))
	plt.xticks(xs,('训练集','验证集','测试集','未知集'),fontproperties='SimHei',
														fontsize=12)
	plt.xlabel('数据集类型',fontproperties='SimHei',fontsize=15)
	plt.ylabel('准确率（%）',fontproperties='SimHei',fontsize=15)
	plt.title('模型准确率柱形图',fontproperties='SimHei',fontsize=16)
	plt.tight_layout()
	plt.show()
	
	
def show_prediction(y_pred,handwriting,origin=True,shuffle=False):
	images,x,y_=handwriting['origin_images'],handwriting['images'],handwriting['labels']
	x=np.where(x>0,1,0)
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
		fig,ax=fig,ax=plt.subplots(5,10,figsize=(10,5))
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


def main(argv=None):
	try:
		mnist=input_data.read_data_sets('../data/MNIST_compressed',one_hot=True)
		handwriting=load_handwriting('../data/my_handwriting')
		acc,y_pred_mydata=calculate_accuracy(mnist,handwriting)
	except NoDataError as err:
		print(err)
	else:
		show_accuracy(acc)
		show_prediction(y_pred_mydata,handwriting)



if __name__=='__main__':
	tf.app.run()
	
