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

import model_utils as mutils
from keras.models import Sequential, Model
from keras.layers import Dense, Cropping2D, Convolution2D, Flatten, Activation, MaxPooling2D, Dropout, Lambda, ELU
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot 
from keras.backend import tf as ktf

#Function to load the existing VGG16 model provided by Keras
#We remove the top FC layers to replace by our own layers
#We will only train the last FC layers
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
import numpy as np

def define_VGG16(fine_tune=False, train_conv_layers=0):
    #The minimum shape for VGG16 is 200x200x3
    #Get VGG Base Model
    pre_process_model = Sequential()
    pre_process_model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    pre_process_model.add(Lambda(lambda X: ktf.image.resize_images(X, (64, 64))))
    pre_process_model.add(Lambda(lambda x: x/127.5 - 1.))
    
    base_model = VGG16(input_tensor=pre_process_model.output, weights='imagenet', include_top=False, input_shape=(320, 250, 3))
    features = base_model.output
    
    #Create a Fully Connected Model
    flat1 = Flatten()(features)
    fc1 = Dense(256, activation='relu')(flat1)
    dp1 = Dropout(0.5)(fc1)
    fc2 = Dense(256, activation='relu')(dp1)
    dp2 = Dropout(0.5)(fc2)
    prediction = Dense(1, activation='relu')(dp2)
    
    model = Model(input=base_model.input, output=prediction)
    
    if fine_tune:
        #freeze training only for n-1 conv layers. We will train some conv, and FC part
        for layer in base_model.layers[0:-train_conv_layers]:
            layer.trainable = False
    else:
        #freeze training for base model first
        for layer in base_model.layers:
            layer.trainable = False
            
    return model