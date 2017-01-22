## Project: Behavioral Cloning 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
### Overview

In this project a CNN model was built to simulate a human driving steering inputs to the vehicles. 


### Image Generator
To generate images for training i used Udacity samples that were provided on the course website. 
The images for training are randomly selected in two step. First a row in the driving_log.csv is 
randomly selected. Then one picture among the left,right,center images is also randomly selected.
The response (steer angle) of left images are adjusted by a configurable value (steerAdjustment).
The number of samples (batch size) is also a configurable parameter of the image generator.  


## Sample Images 
![Sample Image 1](https://github.com/Sadrpour/UdacityProjetcs/raw/master/P3-DriverClonning/right_2016_12_01_13_43_20_478.jpg)
![Sample Image 1](https://github.com/Sadrpour/UdacityProjetcs/raw/master/P3-DriverClonning/right_2016_12_01_13_45_00_946.jpg)
![Sample Image 1](https://github.com/Sadrpour/UdacityProjetcs/raw/master/P3-DriverClonning/right_2016_12_01_13_45_54_661.jpg)
![Sample Image 1](https://github.com/Sadrpour/UdacityProjetcs/raw/master/P3-DriverClonning/right_2017_01_13_18_27_19_925.jpg)


### Model structure
to reduce the number of parameters the incoming images are (1) rescaled into a smaller size (2)
cropped from the top. The rescaling of the image is incoporated within the first layer of the model.
This is achieved by an average-pooling layer. The reason why average-pooling was chosen was intuition.
I felt taking averages of pixels (rather than their max values) will provide a more accurate representation
of the image in a lower resolution. 
The top 60 pixels are mainly the sky and vegetation background and not part of the road. They were removed
via a configurable parameter called crop. 

During trial and error with models, i noticed we can achieve reasonable driving with even tiny models
(less than 200 variables with 1 convolutional layer). However, i also noticed that the performance was very
sensitive to the training set. For example, the car would behave as expected in one trial, and then
would behave quite differently in a different trial (trial = training the model again on the same training set).
As i increased the size of the model, i was able to achieve more stable performance while keeping the number of 
parameters small. 

Drop out was used to prevent over-fitting. 5% of the dataset was set aside for validation. a small regularization 
was also applied to convolutional layers. 
Below is the summary of the model. The idea was to give it flexibility while keeping number of parameters low.


## Model
![Model Architecture](https://github.com/Sadrpour/UdacityProjetcs/raw/master/P3-DriverClonning/model.png)
![Model Training loss](https://github.com/Sadrpour/UdacityProjetcs/raw/master/P3-DriverClonning/training.png)

[![ScreenShot](test run)](https://github.com/Sadrpour/UdacityProjetcs/raw/master/P3-DriverClonning/desktop-animation.gif)

### Tests
The model was tested on the first track, and was observed to remain within the road. 
The model fails on the second track, and i will working on improving that time permitting. 


### notes for my personal understanding 

lets assume i have:
10000 training samples
gen = data_generator(training_data, batch_size= 32)
fit_generator(gen , samples_per_epoch = 100, nb_epoch = 10)
the following will happen
32 samples are drawn from the data_generator. the model updates its parameters based on these 32 samples, we are still in epoch 1 because we have not hit 100 training samples yet.
3 other 32 samples are generated from the generator and for each the them, the above process is repeated. The total samples now stands at 4x32 = 128, which is beyond 100 samples set by samples_per_epoch. Keras may complain about this that it has seen 28 extra samples and was expecting to only see 100, but won't stop moving forward.
at this point, we will go to the next epoch. if we have given keras a validation set, that is when it will provide statistics about how the current model does on validation set.

training continues. in each epoch 128 samples are generated (by running the generator 4 times), model is updated 4 times in each epoch. So in total model has seen 128x10 training samples. In that sense,
samples_per_epoch controls when it is time to go to the next epoch. 
epoch controls (1) the total number of training samples which is approximately equal to samples_per_epoch times the nb_epoch (note this is unrelated to the size of the training data which is 10000) (2) when the model in evaluated by validation dataset. 
