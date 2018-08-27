import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import random
import time
import layers




def uNet(input_shape=None, layers=3, stride=1, pool=2, learn_rate=1.0e-4, epochs=10e4, train_size=3, input_shape=None, 
	channels=3, classes=2, filter=5):

	features = [channels,32,64,128,256,512,800,classes]
	x = tf.placeholder(tf.float32, [None,None,None,features[0]])
	y_ = tf.placeholder(tf.int32, [None,None,None,1])

	# Creating wegihts for each layer and biases
	W_conv1 = layers.weight_variable([filter,filter,features[0], features[1]])
	b_conv1 = layers.bias_variable([features[1]])

	W_conv1 = layers.weight_variable([filter,filter,features[1], features[2]])
	b_conv1 = layers.bias_variable([features[2]])

	W_conv1 = layers.weight_variable([filter,filter,features[2], features[3]])
	b_conv1 = layers.bias_variable([features[3]])

	W_conv1 = layers.weight_variable([filter,filter,features[3], features[4]])
	b_conv1 = layers.bias_variable([features[4]])

	W_conv1 = layers.weight_variable([filter,filter,features[4], features[5]])
	b_conv1 = layers.bias_variable([features[5]])



	# Block 1
	
	