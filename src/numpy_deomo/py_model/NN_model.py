'''
用 python 定义一个前馈神经网络的类
'''


import time
import numpy as np


class NeuralNet:
	def __init__(
				self,input_mode,output_mode,layer_mode=30,l1=0,l2=0,
				training_steps=500,learning_rate=0.01,n_batches=1,
				learn_rate_decay=0,alpha=0,is_shuffle=True,random_state=None
				):
		np.random.seed(random_state)
		self.input_mode=input_mode
		self.output_mode=output_mode
		self.layer_mode=layer_mode
		self.l1=l1
		self.l2=l2
		self.alpha=alpha
		self.training_steps=training_steps
		self.learning_rate=learning_rate
		self.learn_rate_decay=learn_rate_decay
		self.is_shuffle=is_shuffle
		self.n_batches=n_batches	
		self._initialize_w_b()

	def train(self,x,y_,is_print=True):
		self.cost_=[]
		delta_prev={key:np.zeros(param.shape) for key,param in zip(
						('w1','w2','b1','b2'),(self.w1,self.w2,self.b1,self.b2))}
		for step in range(1,self.training_steps+1):
			self.learning_rate/=(1+self.learn_rate_decay*step)
			if self.is_shuffle:
				index=np.random.permutation(y_.shape[0])
				sx,sy_=x[index],y_[index]
			mini=np.array_split(range(y_.shape[0]),self.n_batches)
			cost=0
			for idx in mini:
				mx,my_=sx[idx],sy_[idx]
				output,layer=self._forward_propagation(mx)
				delta_prev=self._back_propagation(mx,my_,output,layer,delta_prev)
				cost+=self._get_cost(my_,output)
			self.cost_.append(cost/self.n_batches)
			if is_print:
				self._print_progress(step)
		return self
				
	def predict(self,input):
		output,layer=self._forward_propagation(input)
		y_pred=np.argmax(output,axis=1)
		return y_pred		
		
	def _initialize_w_b(self):
		self.w1=np.random.randn(self.input_mode,self.layer_mode)
		self.w2=np.random.randn(self.layer_mode,self.output_mode)
		self.b1=np.random.randn(1,self.layer_mode)
		self.b2=np.random.randn(1,self.output_mode)
		
	def _activation(self,z):
		z=np.clip(z,-36,36)
		return 1/(1+np.exp(-z))
		
	def _activation_grad(self,a):
		return a*(1-a)
		
	def _forward_propagation(self,x):
		layer=self._activation(np.dot(x,self.w1)+self.b1)
		output=self._activation(np.dot(layer,self.w2)+self.b2)
		return output,layer

	def _back_propagation(self,x,y_,output,layer,delta_prev):
		grad_w1,grad_w2,grad_b1,grad_b2=self._get_gradient(
											x,y_,output,layer)
		grad_rel_w1=self.l2*self.w1+self.l1*np.sign(self.w1)
		grad_rel_w2=self.l2*self.w2+self.l1*np.sign(self.w2)
		delta_w1=self.learning_rate*(grad_w1+grad_rel_w1)
		delta_w2=self.learning_rate*(grad_w2+grad_rel_w2)
		delta_b1=self.learning_rate*grad_b1
		delta_b2=self.learning_rate*grad_b2
		self.w1-=delta_w1+self.alpha*delta_prev['w1']
		self.w2-=delta_w2+self.alpha*delta_prev['w2']
		self.b1-=delta_b1+self.alpha*delta_prev['b1']
		self.b2-=delta_b2+self.alpha*delta_prev['b2']
		return {'w1':delta_w1,'w2':delta_w2,'b1':delta_b1,'b2':delta_b2}
		
	def _l2_reg(self):
		return (self.l2/2)*(np.square(self.w1).sum()+np.square(self.w2).sum())
		
	def _l1_reg(self):
		return (self.l1/2)*(np.abs(self.w1).sum()+np.abs(self.w2).sum())
		
	def _get_cost(self,y_,output):
		loss=-np.sum(y_*(np.log(output))+(1-y_)*np.log(1-output))
		regularization=self._l1_reg()+self._l2_reg()
		cost=loss+regularization
		return cost
		
	def _get_gradient(self,x,y_,output,layer):
		output_error=output-y_
		layer_error=np.dot(output_error,self.w2.T)*self._activation_grad(layer)
		grad_w1=np.dot(x.T,layer_error)
		grad_w2=np.dot(layer.T,output_error)
		grad_b1=np.sum(layer_error,axis=0,keepdims=True)
		grad_b2=np.sum(output_error,axis=0,keepdims=True)
		return grad_w1,grad_w2,grad_b1,grad_b2
		
	def _print_progress(self,step):
		if step==1:
			self.time=time.clock()
			print('--------训练开始--------')
		c=step/self.training_steps
		count=int(c*100)
		rest=100-count
		a,b='*'*(count),'.'*(rest)
		interval=time.clock()-self.time
		print('{:6.1%} [{}-->{}] {:.0f}秒\r'.format(c,a,b,interval),end='')
		if step==self.training_steps:
			print('\n--------训练结束--------')
			self.time=time.clock()-self.time 
