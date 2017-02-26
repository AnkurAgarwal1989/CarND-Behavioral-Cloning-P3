#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:03:36 2017
@author: ankur
"""

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

#Function takes a vector and generates a 1D Gaussian Signal, with mu and sigma
def gauss_1D(x, mu, sigma):
    return np.exp(-(x - float(mu))**2 / (2*float(sigma)**2))

#Function to generate 2D Gaussian Kernel given a mean, sigma
def gauss_kernel(kernel_size, mu, sigma):
    kernel_size = max(3, kernel_size)
    x = np.linspace(-kernel_size, kernel_size, kernel_size)
    x_g = gauss_1D(x, mu, sigma)
    
    kernel = np.outer(np.transpose(x_g), x_g)
    kernel = kernel / np.sum(kernel)
    return kernel

##Layer to implement Gaussian Blurring on 3 channel RGB images
#The layer does not have trainable weights. Weights are fixed to a Gaussian Kernel
#Layer takes an input image and returns a blurred 3channel images
#Arguments: kernel_size: size of gaussian kernel to be created
#           weights: weights for the Gaussian filter
class GaussBlurLayer(Layer):
    def __init__(self, kernel_size, weights, **kwargs):
		self.initial_weights = weights
        super(GaussBlurLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(shape=(kernel_size, kernel_size),
                                 initializer=self.initial_weights,
                                 trainable=False)
        super(GaussBlurLayer, self).build()  # Be sure to call this somewhere!

    def call(self, x, mask=None):
		##Split data along 3rd axis to do 2D convolution for gaussian blurring
		img = K.split(x, 3, axis=3)
		
		for img_c in img:
			img_c = K.conv2d(img_c, self.W, border_mode='same')
		x = K.concat(img, axis=3)
		return x

    def get_output_shape_for(self, input_shape):
        return input_shape