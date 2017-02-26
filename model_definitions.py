#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:025:17 2017
@author: ankur
"""

'''
Functions to define the VGG and NVIDIA end-to-end learning architectures
VGG16 is used for Fine-tuning
NVIDIA is used for End-to-End learning
'''

import model_utils
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, Activation, MaxPooling2D, Dropout, Lambda, ELU
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot 

#Function to load the existing VGG16 model provided by Keras
#We remove the top FC layers to replace by our own layers
#We will only train the last FC layers
def VGG16():
	model = Sequential()
	return model

	
#Function to define a fresh NVIDIA model.
#We will train this model end-to-end
def NVIDIA():
	model = Sequential()
	return model