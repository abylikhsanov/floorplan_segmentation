import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pylab import *
from keras_contrib.applications import densenet
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer



def uNet(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):


	# Block 1
	x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
	x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
	x = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(x)

	# Block 2
	x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
	x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
	