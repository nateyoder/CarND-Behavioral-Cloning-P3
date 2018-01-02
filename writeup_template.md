# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5, 3x5, or 3x3 filter sizes and 
depths between 24 and 64 (model.py lines 54-71) 

The model includes RELU layers to introduce nonlinearities for the CNN layers. 
The input data is normalized in the model using a Keras lambda layer (code line 56) 
and cropped using a Keras cropping layer (code line 55). 

#### 2. Attempts to reduce overfitting in the model
The model contains dropout layers between the dense layers in order to reduce overfitting 
(model.py lines 63, 65, 67 and 69). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 
51-52). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and 
recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to investigate the model used by NVIDIA in their 
self-driving car [paper](https://arxiv.org/pdf/1604.07316.pdf)

My first step was to use a series of convolution neural networks to analyze the images and extract features from the
 video frames.  This sort of layer CNN feature extraction is similar to what is used for image classification.
This information was then fed to several dense layers of smaller and smaller size to gradual reduce the information 
to output a single variable which was the desired steering angle.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. 
This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it included dropout layers between the dense layers.

The final step was to run the simulator to see how well the car was 
driving around track one.
It seemed to have some issues making the turn after the bridge 
so I collected some more training data around this location and retrained the model.

At the end of the process, the vehicle is able to drive autonomously around the 
track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 54-69) consisted of five
 convolution layers followed by four dense layers.  
 The dense layers were separated by Dropout layers to reduce overfitting.

Here is a visualization of the architecture with the sizes specified:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the 
vehicle would learn to recover properly.  I made sure to do this from the side as well as when the car was heading 
toward the side.

To augment the data sat, I also flipped images horizontally and negated the driving angles.
This helps effectively double the training data and compensates 
for the fact that the track turns mostly in a single direction because it is a loop.

After the collection process, I had 39,738 frames of data. Each frame of data was normalized and input into 
the network. 

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was not determined as the validation loss continued 
to decrease throughout the training process. 
However, given the performance of the developed model 
10 epochs was determined to be sufficient
training was stopped at this point to save time. 
I used an adam optimizer so that manually training the learning rate was not necessary.
