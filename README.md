[//]: # (Image References)
[image1]: ./images/cnn.png "CNN Architecture"

# Behaviorial Cloning Project

Overview
---
This repository contains starting files for the Behavioral Cloning Project.


The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

Architecture
---
For the project I used a vanilla CNN, whose architecture is based on one [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) from Nvidia. The only modification I did was to add one layer, after normalization, to crop the images from the virtual cam and get the superior part removed as non relevant for the task.  Here's the CNN:

![cnn architecture][image1] 

This is an end-to-end solution, meaning that the CNN, from the image stream, it infer directly the steering commands. 


### Training
For training the CNN I drove the simulated car and logged the telemetrics (and images from the a virtual cam on the windscreen) on disk. The data is loaded, augmented and split in 80-20 for training and validation. For more details, see the writeup.