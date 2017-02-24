#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:56:00 2017

@author: ankur
"""

import os
import csv
import cv2
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

from data_generator import generator

##utility function to plot data

def plot_image_data(train_batch_X, train_batch_y):
    gs = gridspec.GridSpec(len(train_batch_y)//3, 3, hspace = 0.5, wspace=0)
    plt.figure(figsize=(10, len(train_batch_y)*1.5//3))

    for i in range(len(train_batch_X)):
        ax = plt.subplot(gs[i])
        #ax.tick_params('off')
        ax.imshow(cv2.cvtColor(train_batch_X[i], cv2.COLOR_BGR2RGB))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        title = train_batch_y[i]
        ax.set_title(title)
    plt.axis('off')
    plt.suptitle("Sample Images from generator")  
    plt.show()
    
    
data_dir = '../BehavClone_training';
if not os.path.isdir(data_dir):
    print("data directory doesn't exist")

csv_file = os.path.join(data_dir, 'driving_log.csv')
if not os.path.isfile(csv_file):
    print("Could not find CSV file")

image_dir = os.path.join(data_dir, 'IMG')
if not os.path.isdir(image_dir):
    print("Could not find image directory")
    
samples = []
with open(csv_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=5)
validation_generator = generator(validation_samples, batch_size=5)
train_batch_X, train_batch_y = next(train_generator)
plot_image_data(train_batch_X, train_batch_y)