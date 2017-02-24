#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:53:36 2017

@author: ankur
"""

import cv2
import numpy as np
from sklearn.utils import shuffle

##Function to apply affine transform (translation, rotation and skew), flipping
# Input: imageS: opencv style
#        affine_jitter: jitter limit in pixels 
#Output: returns a list of augmented images
def augment_data(images, angles, affine_jitter=0):
    flip_images, flip_angles = flip_data(images, angles)
    images = np.vstack((images, flip_images));
    angles = np.hstack((angles, flip_angles));
    
    return images, angles

def flip_data(images, angles):
    flip_images = images[:, :, ::-1, :];
    flip_angles = -angles;
    return flip_images, flip_angles
    
    
def jitter(images, angles):
    return

##Function generates batch_size of data from the samples provided
#Input: samples: list of samples from the data .csv
#       batch_size: size of generated data (4th dim)

def generator(samples, batch_size=32):
    correction = 0.2;
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                image_name = batch_sample[0]
                image = cv2.imread(image_name)
                angle = float(batch_sample[3])
                images.append(image)
                angles.append(angle)
                
                #[(left_img_idx, +ve correction), (right_img_idx, -ve correction)]
                for idx, sign in [(1, 1), (2, -1)]:
                    image_name = batch_sample[idx]
                    image = cv2.imread(image_name)
                    images.append(image)
                    angles.append(angle + (sign*correction))
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            X_train, y_train = augment_data(X_train, y_train)
            
            yield shuffle(X_train, y_train)