# Behavioral Cloning Project
___

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./images/cnn.png "CNN" 
[image2]: ./images/model_loss.png "CNN"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network whose architecture was taken as is from the Nvidia [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

The data is normalized in the model using a Keras lambda layer (model.py line 103). Also, to increase effectiveness of the training data, I removed non-relevant part of the images from the virtual dash camera, by cropping them. (line 108) 

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting I simply used data augmentation.  Keras comes with an utility class called `ImageDataGenerator` and I used that to slightly rotate and shift the images (model.py line 169) 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 93). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, but I used code to have the learning rate customizable (model.py line 154) but in the end I didnt touch it other than setting the first time, after a quick look at the model loss graph.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. At the beginning I wanted to use a combination of center left and right sides of the road, by using a weighted combination based on probability (model.py line 185) but during my experiments the right and left images didn't improve the training, even with different ranges of angle adjustement. At the end I decided to go only with the center camera 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to following the suggestions from the lessons. An Nvidia paper was released about end-to-end solution. This seems a really good starting point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. At the beginning overfitting was not my problem as I started directly with data-augmentation. I faced another problem instead: the training error was greater than the validation, with the later going up and down. See the figure:

![cnn][image2]

I spent several hours fiddling with code to improve things without success. I spoke with my tutor and he asked me if I had a look at the histogram for steering angle distribution. Well, I forgot to do so, so I wrote the code to redistribute the training set with equal probability for given slots of angle values. See model.py lines 19-48.  

After normalization of distribution and with pre-existing data augmentation I managed to get good results. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.  I could drive the simulator also in the second track and I believe I could get even better results but I had to wrap-up for time lack.

#### 2. Final Model Architecture

The final model architecture (model.py lines 96-158) consisted of a convolution neural network with the following layers and layer sizes: 

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 105, 320, 3)   0           lambda_1[0][0]                   
____________________________________________________________________________________________________
conv1 (Convolution2D)            (None, 51, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
conv2 (Convolution2D)            (None, 24, 77, 36)    21636       conv1[0][0]                      
____________________________________________________________________________________________________
conv3 (Convolution2D)            (None, 10, 37, 48)    43248       conv2[0][0]                      
____________________________________________________________________________________________________
conv4 (Convolution2D)            (None, 8, 35, 64)     27712       conv3[0][0]                      
____________________________________________________________________________________________________
conv5 (Convolution2D)            (None, 6, 33, 64)     36928       conv4[0][0]                      
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 12672)         0           conv5[0][0]                      
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           1267300     flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 1,404,219
Trainable params: 1,404,219
Non-trainable params: 0
```

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![cnn][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Then I recorded two laps going in the opposite direction, to have steering angles also in the other direction. After that, I tried to drive the car close to the margins of the roads, activate the recording and then move away from the side toward the center of the road. That's all, I didnt drive the other circuit. 
