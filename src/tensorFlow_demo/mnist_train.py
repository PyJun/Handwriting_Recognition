'''
小批次梯度下降法反向传播训练神网络
'''


import os
import time
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tutorials.mnist import input_data
from py_model.mnist_inference import *


BATCH_SIZE=100
LEARNING_RATE_BASE=0.6
LEARNING_RATE_DECAY=0.99
REGULARIZER_RATE=0.0005
TRAINING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99

FILEPATH='../data/model/tf'


def print_progress(step,epochs,start):
	if step==1:
		print('\n--------训练开始--------')
	count=int(step/epochs*100)
	rest=100-count
	a,b='*'*(count),'.'*(rest)
	c=step/epochs
	interval=time.clock()-start
	print('{:6.1%} [{}-->{}] {:.0f}秒\r'.format(c,a,b,interval),end='')
	if step==epochs:
		print('\n--------训练结束--------')	


def show_cost(cost):
	plt.figure(figsize=(8,5))
	plt.plot(range(len(cost)),cost)
	plt.ylim([0,1])
	plt.ylabel('Cost')
	plt.xlabel('Step')
	plt.title('损失函数曲线',fontproperties='SimHei',fontsize=16)
	plt.tight_layout()
	plt.show()

	
def train(mnist,filename='NN.ckpt'):
	x=tf.placeholder(tf.float32,[None,INPUT_MODE],name='x-input')
	y_=tf.placeholder(tf.float32,[None,OUTPUT_MODE],name='y-input')
	regularizer=tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
	y=inference(x,regularizer)
	global_step=tf.Variable(0,trainable=False)
	variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
	variable_averages_op=variable_averages.apply(tf.trainable_variables())
	cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(
					logits=y,
					labels=tf.argmax(y_,axis=1)
					)
	cross_entropy_mean=tf.reduce_mean(cross_entropy)
	learning_rate=tf.train.exponential_decay(
					LEARNING_RATE_BASE,
					global_step,
					mnist.train.num_examples//100,
					LEARNING_RATE_DECAY
					)
	loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
	train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(
												loss,global_step=global_step
												)
	train_op=tf.group(train_step,variable_averages_op)

	saver=tf.train.Saver()
	
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		start=time.clock()
		cost=[]
		for i in range(TRAINING_STEPS):
			xs,ys=mnist.train.next_batch(BATCH_SIZE)
			xs=np.where(xs>0.4,1,0)
			_,cost_,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
			cost.append(cost_)
			if step%100==0:
				print_progress(step//100,TRAINING_STEPS//100,start)
		saver.save(sess,os.path.join(FILEPATH,filename))
		show_cost(cost)
				
		
def main(argv=None):
	mnist=input_data.read_data_sets(os.path.join('../data/MNIST_compressed'),one_hot=True)
	train(mnist)
	input()


if __name__=='__main__':
	tf.app.run()
