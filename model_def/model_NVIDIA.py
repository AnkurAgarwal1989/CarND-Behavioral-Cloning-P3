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

from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Flatten, Activation, MaxPooling2D, Dropout, Lambda, ELU, Cropping2D
from keras.backend import tf as ktf

#Function to define a fresh NVIDIA model.
#We will train this model end-to-end
def define_NVIDIA(image_height=64, image_width=64, channels=3):
    model = Sequential()
    #model.add(Cropping2D(cropping=((50,20), (0,0)),
                         #input_shape=(image_height,image_width,channels)))
    #model.add(Lambda(lambda X: ktf.image.resize_images(X, (64, 64))))
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(image_height,image_width,channels)))
    
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Activation('elu'))
    model.add(Dense(1164))
    model.add(Dropout(.5))
    model.add(Activation('elu'))
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(Activation('elu'))
    model.add(Dense(50))
    model.add(Dropout(.5))
    model.add(Activation('elu'))
    model.add(Dense(10))
    model.add(Dropout(.5))
    model.add(Activation('elu'))
    model.add(Dense(1))
    
    return model