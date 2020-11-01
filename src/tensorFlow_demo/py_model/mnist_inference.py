'''
定义神经网络的前向传播
'''


# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


INPUT_MODE=784
OUTPUT_MODE=10
LAYER1_MODE=500


def get_weight_variable(shape,regularizer=None):
	weights=tf.get_variable('weights',shape,initializer=
							tf.truncated_normal_initializer(stddev=0.1))
	
	if regularizer !=None:
		tf.add_to_collection('losses',regularizer(weights))
		
	return weights
	
	
def inference(input_tensor,regularizer):
	with tf.variable_scope('layer1'):
		weights=get_weight_variable([INPUT_MODE,LAYER1_MODE],regularizer)
		biases=tf.get_variable('biases',[LAYER1_MODE],
								initializer=tf.zeros_initializer())
		layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
		
	with tf.variable_scope('layer2'):
		weights=get_weight_variable([LAYER1_MODE,OUTPUT_MODE],regularizer)
		biases=tf.get_variable('biases',[OUTPUT_MODE],initializer=
									tf.zeros_initializer())
		layer2=tf.matmul(layer1,weights)+biases
		
	return layer2
