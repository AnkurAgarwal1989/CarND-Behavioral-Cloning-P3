#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:03:36 2017
@author: ankur
"""

from keras import backend as K
from keras.engine.topology import Layer
import functools
from keras import initializations

##Layer to implement Gaussian Blurring on 3 channel RGB images
#The layer does not have trainable weights. Weights are fixed to a Gaussian Kernel
#Layer takes an input image and returns a blurred 3channel images
#Arguments: kernel_size: size of gaussian kernel to be created
#           weights: weights for the Gaussian filter
class GaussBlurLayer(Layer):
    def __init__(self, kernel_size, weights, init='glorot_uniform', dim_ordering='default', **kwargs):
        self.kernel_size = kernel_size
        self.initial_weights = weights
        self.init = initializations.get(init)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.dim_ordering=dim_ordering
        #self.input_shape = input_shape
        #self.batch_input_shape = batch_input_shape
        super(GaussBlurLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        #self.W = self.initial_weights
        self.W = self.add_weight((self.kernel_size, self.kernel_size, 1, 1),
                                 initializer=functools.partial(self.init,
                                 dim_ordering='tf'),
                                 trainable=False)
        print(len(self.get_weights()))
        print(self.initial_weights.shape)
        self.set_weights([self.initial_weights])
        
        super(GaussBlurLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        
        ##Split data along 3rd axis to do 2D convolution for gaussian blurring
#        img = K.split(x, 3, axis=3)
#        for img_c in img:
#            img_c = K.conv2d(img_c, self.W, border_mode='same')
#        x = K.concat(img, axis=3)
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape