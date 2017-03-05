#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from model_def import model_NVIDIA
import model_utils as mutils
import argparse
import os
from time import gmtime, strftime

import csv
from data_generator import generator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
# ## Image Augmentation
# All of this goes in the Generator Function
# 1. crop
# 2. Normalize
# 2. gaussian blur
# 3. augmentation: translating, jitter
# 3. resize
# 4. augmentation: flip

# ## Data Generator
# We want to generate (and augment) data on the fly

def main():
    
    parser = argparse.ArgumentParser(description='Train NVIDIA End-to-End Learning model')
    parser.add_argument('--init', help="Path to .h5 file (optional). If not provided, a new model will be created and trained", type=str)
    parser.add_argument('--save', help="Path to save .h5 file. If not provided a generic timestamped name will be used and saved", type=str)
    
    args = parser.parse_args()
    init_file=args.init
    save_file=args.save
    if save_file is None:
        timestamp=strftime("%Y-%m-%d_%H:%M", gmtime())
        out_name =timestamp+'_model.h5';
        save_file = os.path.join(out_name);
    
    if init_file is None:
        model = model_NVIDIA.define_NVIDIA()
        len(model.layers)
    else:
        model=mutils.load_net(init_file)
        if model is None:
            print('Could not find right model. Exiting')
            exit
    
    print(save_file)
    print(init_file)
    samples = []
    data_dir = ['../BehavClone_training'] ## '../BehavClone_training']#, './'];
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
    
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
        
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-3))
    model_checkpoint = ModelCheckpoint(filepath='model.weights.{epoch:02d}-{val_loss:.5f}.h5', verbose=1, save_best_only=True, save_weights_only=True, period=20)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    reduce_LR_plateau = ReduceLROnPlateau(factor=0.1, patience=5, verbose=1, epsilon=1e-5)
    tensorboard_log = TensorBoard(log_dir='./tb_logs', histogram_freq=0, write_graph=True, write_images=False)
    
    batch_size= 15
    augmentation_factor = (3*2)*2 # (Left+right+center) * flip
    samples_per_epoch = len(train_samples)*augmentation_factor
    nb_val_samples = len(validation_samples)*augmentation_factor
    n_epoch = 30
    fit = model.fit_generator(generator(train_samples, batch_size),
                              samples_per_epoch=samples_per_epoch,
                              nb_epoch=n_epoch, verbose=2,
                              callbacks=[model_checkpoint, early_stopping,
                                         reduce_LR_plateau, tensorboard_log],
                              validation_data=generator(validation_samples, batch_size),
                              nb_val_samples = nb_val_samples,
                              initial_epoch = 0)
    print("Done Training")
    mutils.save_net(model, save_file)
#    
    
if __name__ == "__main__":
    main()

