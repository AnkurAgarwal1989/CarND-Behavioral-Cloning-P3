#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 18:06:45 2017
@author: ankur
"""

'''
Utility functions to initiate training, saving and loading models and weights
'''

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.models import load_model, model_from_json, Model, Sequential
from keras.optimizers import Adam
import data_generator as gen
import json
import os

import numpy as np

#using
#ModelCheckpoint to save model every few epochs
#EarlyStopping to finish training if loss doesnt reduce
#ReduceLROnPlateau reduce learning rate on the fly as required
#TensorBoard to generate TensorFlow friendly logs

#if resuming traning, begin_at_epoch will change, else remains 0
def begin_training(model, training_samples, validation_samples, begin_at_epoch=0):
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-3))
    model_checkpoint = ModelCheckpoint(filepath='model.weights.{epoch:02d}-{val_loss:.5f}.h5', verbose=1, save_best_only=True, save_weights_only=True, period=20)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    reduce_LR_plateau = ReduceLROnPlateau(factor=0.1, patience=2, verbose=1, epsilon=1e-5)
    tensorboard_log = TensorBoard(log_dir='./tb_logs', histogram_freq=0, write_graph=True, write_images=False)

    batch_size= 128
    samples_per_epoch = 8000
    nb_val_samples = 2000
    n_epoch = 100
    
    fit = model.fit_generator(gen.generator(training_samples, batch_size),
                              samples_per_epoch=samples_per_epoch,
                              nb_epoch=n_epoch, verbose=2,
                              callbacks=[model_checkpoint, early_stopping,
                                         reduce_LR_plateau, tensorboard_log],
                              validation_data=gen.generator(validation_samples, batch_size),
                              nb_val_samples = nb_val_samples,
                              initial_epoch = begin_at_epoch)
    print("Done Training")
	
	
##Function to save model architecture
def save_model_JSON(model, filename):
    print("Saving model architecture");
    model_json = model.to_json();
    with open (filename, 'w') as json_file:
        json.dump(model_json, json_file, indent=4, sort_keys=True, separators=(',', ':'))
    print("Model saved to ", filename, " in current directory");

##Function to load model architecture
def load_model_JSON(filename):
    if os.path.isfile(filename):
        with open(filename, 'r') as json_file:
            model = model_from_json(json_file.load())
            return model
    return None

##Function to save entire model (architecture+weight)
def save_net(model, filename):
    model.save(filename)  # creates a HDF5 file
    return

# Function to load compiled model(architecture+weight)
def load_net(filename):
    if os.path.isfile(filename):
        model = load_model(filename)
        return model
    return None;

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