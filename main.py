import numpy as np
import tensorflow as tf
import layers

def create_model(x, keep_prob, output_depth, layers=3, input_depth, pool_size=2, summeries=True):
	with tf.name_scope('preprocessing'):
		nx = tf.shape(x)[1]
		ny = tf.shape(x)[2]
		image = tf.reshape(x, tf.stack([-1, nx, ny, input_depth]))
		batch_size = tf.shape(x_image)[0]

	weights = []
	bias = []
	convs = []
	pools = OrderedDict()
	deconv = OrderedDict()
	dw_h_convs = OrderedDict()
	up_h_convs = OrderedDict()
	size = 1000

	# Down Layers

	# Block 1 -> 3:

	for layer in range(0,layers):
		with tf.name_scope("down_conv{}".format(str(layer))):
			features = 2 ** layer * features_root
			sigma

	# Block 2:



