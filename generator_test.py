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
    gs = gridspec.GridSpec(len(train_batch_y)//3, 3, hspace = 0.5, wspace=0.3)
    plt.figure(figsize=(5, len(train_batch_y)*1.5//3))

    for i in range(len(train_batch_X)):
        ax = plt.subplot(gs[i])
        #ax.tick_params('off')
        ax.imshow(cv2.cvtColor(train_batch_X[i], cv2.COLOR_BGR2RGB))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        title = train_batch_y[i]
        ax.set_title(title)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    plt.suptitle("Sample Images from generator")  
    plt.show()
    plt.savefig("generator.png")
    
    
samples = []
data_dir = ['./'];
for training_dir in data_dir:
    if not os.path.isdir(training_dir):
        print("data directory doesn't exist")

    csv_file = os.path.join(training_dir, 'driving_log.csv')
    if not os.path.isfile(csv_file):
        print("Could not find CSV file")

    image_dir = os.path.join(training_dir, 'IMG')
    if not os.path.isdir(image_dir):
        print("Could not find image directory")
    
    print(csv_file)
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2, random_state=1200)
train_samples = train_samples[5:6]
train_generator = generator(train_samples, batch_size=1, drop_prob=0)
plt.figure(figsize=(6, 2))
print(train_samples)
plt.xticks([], [])
plt.yticks([], [])
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(cv2.imread(train_samples[0][1].strip()), cv2.COLOR_BGR2RGB))
s = float(train_samples[0][3])+0.2;
plt.title('Left: {}'.format(str(s)))
plt.xticks([], [])
plt.yticks([], [])
plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(cv2.imread(train_samples[0][0].strip()), cv2.COLOR_BGR2RGB))
s = s-0.2;
plt.title('Center: {}'.format(str(s)))
plt.xticks([], [])
plt.yticks([], [])
plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(cv2.imread(train_samples[0][2].strip()), cv2.COLOR_BGR2RGB))
s = s-0.2;
plt.title('Right: {}'.format(str(s)))
plt.xticks([], [])
plt.yticks([], [])
plt.suptitle("Input to generator")  
plt.show()
plt.savefig("input.png")
validation_generator = generator(validation_samples, batch_size=1)
train_batch_X, train_batch_y = next(train_generator)
plot_image_data(train_batch_X, train_batch_y)