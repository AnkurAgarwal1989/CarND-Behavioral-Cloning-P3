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
                      
    bright_images, bright_angles = add_brightness_noise(images, angles)
    images = np.vstack((images, bright_images));
    angles = np.hstack((angles, bright_angles));
    return images, angles

def flip_data(images, angles):
    flip_images = images[:, :, ::-1, :];
    flip_angles = -angles;
    return flip_images, flip_angles

def add_brightness_noise(images, angles):
    new_images = []
    new_angles = []
    for img, angles in zip(images, angles):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        #Add randowm noise to L channel
        #We are making the images light or dark
        hls[:,:,1] = hls[:,:,1] + np.random.uniform(-30, 30, img.shape[0:-1])
        new_images.append(cv2.cvtColor(hls, cv2.COLOR_HLS2RGB))
        new_angles.append(angles)
    return new_images, new_angles

def crop_image(image):
    image = image[65:-25,:,:]
    return image

def resize_image(image):
    image = cv2.resize(image,(64, 64), interpolation = cv2.INTER_AREA)
    return image

def preproc_image(image):
    return resize_image(crop_image(image))
    
def jitter(images, angles):
    return



##Function generates batch_size of data from the samples provided
#Input: samples: list of samples from the data .csv
#       batch_size: size of generated data (4th dim)

def generator(samples, batch_size=32, drop_prob=0.4):
    correction = 0.2;
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                image_name = batch_sample[0].strip()
                
                image = cv2.imread(image_name)
                
                angle = float(batch_sample[3])
                if angle == 0.0:
                    if np.random.random() < drop_prob:
                        continue
                    
                image = preproc_image(image)
                images.append(image)
                
                angles.append(angle)
                #[(left_img_idx, +ve correction), (right_img_idx, -ve correction)]
                for idx, sign in [(1, 1), (2, -1)]:
                    image_name = batch_sample[idx].strip()
                    image = cv2.imread(image_name)
                    
                    image = preproc_image(image)
                    images.append(image)
                    angles.append(angle + (sign*correction))
            
            if len(images) == 0:
                continue
            
            # trim image to only see section with road
            X_train = np.array(images)
            
            y_train = np.array(angles)
            X_train, y_train = augment_data(X_train, y_train)
            
            yield shuffle(X_train, y_train)