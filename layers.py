import tensorflow as tf

def weight_variable(shape, stddev=0.1, name='weight'):
	initial = tf.truncated_normal(shape, stddev=stddev)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name='bias'):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x,W,b,keep_prob):
	with tf.name_scope('conv2d'):
		conv_2d =tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='VALID')
		conv_2d_b = tf.nn.bias_add(conv_2d, b)
		return tf.nn.dropout(conv_2d_b, keep_prob)

def deconv2d(x, W, stride):
	with tf.name_scope('deconv2d'):
		x_shape = tf.shape(x)
		output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
		return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1,stride,stride,1], padding='VALID')

def max_pool(x,n):
	return tf.nn.max_pool(x, ksize=[1,n,n,1], strides=[1,n,n,1], padding='VALID')

def crop_and_concat(x1,x2):
	with tf.name_scope('crop_and_concat'):
		x1_shape = tf.shape(x1)
		x2_shape = tf.shape(x2)
		offsets = [0, (x1_shape[1]-x2_shape[1]) // 2, (x1_shape[2]-x2_shape[2]) // 2, 0]
		size = [-1, x2_shape[1], x2_shape[2], -1]
		x1_crop = tf.slice(x1,offsets,size)
		return tf.concat([x1_crop,x2], 3)

def cross_entropy(y_, y):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

def pixel_wise_softmax(y):
	with tf.name_scope('pixel_wise_softmax'):
		max_axis = tf.reduce_max(y, axis=3, keepdims=True)
		exponential_map = tf.exp(y - max_axis)
		normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
		return exponential_map / normalize
