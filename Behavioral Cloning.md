# **Behavioral Cloning**

### This write-up describes a deep learning based approach to "teach" a car to drive. The network learns to steer the car from the simulator data provided by the user and mimics the driver's behaviors. We elaborate upon the choice of network, system design and pre-processing and then briefly analyze the results.

---

**Behavioral Cloning Project**

The goals / steps of this project are as following:
* Use the simulator to collect data of good driving behavior.
* Build a convolution neural network, in Keras, that predicts steering angles from images.
* Train and validate the model with a training and validation set.
* Test that the model successfully drives around the track without leaving the road.


[//]: # "Image References"

[model]: ./examples/model.png "Model Visualization"
[lcr]: ./examples/LeftCenterRight.png "LeftRightCenter"
[bnoise]: ./examples/BrightnessNoise.png "BrightnessNoise"
[recoveryGIF]: ./examples/recovery.gif "Recovery"
[smoothGIF]: ./examples/smoothTurn.gif "SmoothTurning"

## Section I

### Code Layout

- **model_def**: folder; contains the definitions of the models I have used.
- **train_NVIDIA.py**: File to create the NVIDIA model and train it. Training can be resumed from an existing model.
- **data_generator.py**: File defines the data generator for use during Training and Validation. The generator also augments the data.
- **model_utils.py**: Utility functions to save model layouts, weights or visualizations  for debugging.
- **drive.py**: Interacts with the simulator.
- **nvidia_e2e.h5**: HDF5 file that holds the trained network.

Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py nvidia_e2e.h5
```
## Section II

### Model Architecture and Training Strategy

#### Architecture and Selection:

The NVIDIA end-to-end learning model has been used. This decision was based on some initial comparison between VGG16 and NVIDIA. An end-to-end trained NVIDIA model performed better than fine-tuning a pre-trained VGG16.

- The selected NVIDIA model has 5 2D Convolution layers, followed by 5 Fully Connected Layers. The final output is a 1x1 value (steering angle). Function `define_NVIDIA()` in `model_def\model_NVIDIA.py`.

- Convolution layers are a mixture of 3x3 and 5x5 convolutions. The filter depths vary from 24 to 64.

- Each Convolution Layer is followed by a (2x2) Max Pooling operation.

- Each Dense (FC) layer is followed by (0.5) dropout.

- The activation in the network is Exponential Linear Unit (ELU). The ELU non-linearity is a modified version of ReLU and helps prevents activations from dying (becoming 0).

- The data is normalized to [-1, 1] using a Keras Lambda layer.

  TODO: Modelimage

    ![][model]

  ​


#### Reducing Overfitting:

- The network loss is Mean Squared Error of the steering angle. The error metric is not a good measure of accuracy or fit. *Over fitting or Under fitting does not show up during training*; the network needs to be run on the simulator to actually see how it is performing. 
- Extensive use of Dropout layers has been made to reduce the expected over fitting. Each Dense layer is followed by a Dropout layer. The dropout is done with 50% probability.
- L2 Regularization has been used to reduce over fitting, which is a general good practice for training.

#### Reducing Bias:

- Since the car usually drives straight (steering angles of 0), the network showed a bias towards not steering. The data was pruned in data generator by dropping samples with steering angle of 0 with a probability of 40%. Lines 80-82 in `generator()` in `data_generator.py` 


#### Parameter Tuning:

- The model used an ADAM optimizer, so the learning rate was not tuned manually (line 76 in `main() in train_NVIDIA.py`).
- Since our regression output is only one value, during training loss reduces very rapidly. Lines 78-81 in `main() in train_NVIDIA.py` showcase use of *Early Stopping* and *Learning Rate reduction on plateaus*. The values used are found empirically.

### Training:

- Data was randomly shuffled for batch generation and then shuffled after augmentation.
- Training was done on GTX 1080 GPU. Due to Early stopping, training was done for 18 epochs after which no change in loss was seen.
- Training loss and validation loss do not seem to be a good indicator of performance. The true test of a successful model is the car driving being able to drive within the lane and recover when moved to the sides.


## Section III

### Data

#### Collection:

Because this is behavioral cloning and it's not very feasible (or required) to collect millions of data points, the data collection needs to be done strategically.

To model the car to drive like me, I have only used data I collected myself using the simulator provided by Udacity.

- I drove 2 laps, one slow speed and one fast (but smooth) in one direction. I also collected 1 lap driving in the opposite direction to get more viewpoints of the race track.
  - With just this data (~30k), the car does not do a good job at recovery. It can keep driving in the center, but if it deviates, it can not correct itself.
- I then added some recovery data by driving to the edge or the sides and recording only when I steer the car back.
- I also added some data with smoother turns.
- The total data is **~45k samples**. Each sample includes an image from Center, Left and Right camera and a steering angle.
- I used a mouse to steer as the keyboard did not have a good resolution.
- This data has a lot of 0 steering angle images which were dropped as explained in Section II.
- Data was split into Train-Validation sets with an 80-20 split.

#### Augmentation and pre-processsing:

Since the data is images of the driving lane as seen form the car, we can use some valid assumptions and augment our data. All code for this can be found in `generator() in data_generator.py`.

- Flipping the image (about vertical axis) is still a valid driving lane. We just invert the sign of the steering angle.
  - ![][flip]
- The left and right camera images can be thought of as the center camera when the car has steered to one side. This allows us to use the left and right camera images, by adding some compensation to the steering angle. I used a compensation of 0.2 degrees (normalized range [-1, +1]).
  - ![][lcr]

With just this augmentation, I noticed the car did not behave correctly over the bridge or near muddy patches of road. I then decided to perturb the brightness of the image randomly.

- The *L channel* is extracted from the HLS colorspace. A random noise is added to every pixel and the image is converted back to RGB. This has the effect of reducing dependency on color and smoothness of texture (The bridge has a different texture from the road)
  - ![][bnoise]

With these augmentations, each sample yields 12 samples.

All images are cropped (65 pixels from top and 25 pixels from bottom) to remove the sky and the hood of the car. Images are then resized to 64x64.



## Section IV

### Results

- A video clip of the car completing a lap is attached.
- The car stays within the lanes and also recovers when moved away from center.
  - TODO: Add video
- The car turns very smoothly.
  - TODO: Video of Track 1 turning