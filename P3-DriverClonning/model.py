from __future__ import division
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.layers import Conv2D, Flatten
from keras.models import Sequential
from keras.layers import Dropout, Lambda, Convolution2D, ELU, Reshape
from keras.regularizers import l2, activity_l2
from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation, AveragePooling2D 
import gc

flags = tf.app.flags
FLAGS = flags.FLAGS
loc = '/media/pemfir/Data/UdacityProjects/Data/DriverClonning/data/' # location of my data files 

# command line flags
flags.DEFINE_string('training_file', loc + 'driving_log.csv', "training file (.p)")
flags.DEFINE_integer('epoch', 15, "number of epoches")
flags.DEFINE_integer('batchSize', 100, "batch size")
flags.DEFINE_integer('cropSize', 60, "number of pixels to cut from top (add this to drive.py too")
flags.DEFINE_float('steerAdjustment', 0.25, "left and right image steer adjustments")
flags.DEFINE_float('oversamplingCoef', 0.8 , "ratio of nonezero steering to zero steering images")

# removal of old model files 
try:
    os.remove('model.json')
    os.remove('model.h5')
    print('deleted the old model files')
except Exception:
    print('could not find the files to delet ...')

# breaking the training data into training and validation set

df = pd.read_csv(FLAGS.training_file)
df, dfValidation = train_test_split(df , test_size=0.05, random_state=100)


# image generator 
def generate_data(pos,
                  im,
                  df1,
                  df0,
                  steerAdjustment,
                  crop,
                  oversamplingCoef,
                  batchSize):
    while(True):
        # samples with 0 steering were seperated from samples with nonzero steering
        # Also, i over-sampled images with nonzero steering 
        indices1 = np.random.choice(len(df1), int(oversamplingCoef*batchSize))
        indices2 = np.random.choice(len(df0), int((1-oversamplingCoef)**batchSize))
        df3 = df1.iloc[indices1].append(df0.iloc[indices2])
        # removing the top 60 pixels from the incoming images 
        xBatch = np.zeros((batchSize, im.shape[0]-crop, im.shape[1], im.shape[2]), dtype=np.float32)
        yBatch = np.zeros((batchSize,), dtype=np.float32)
        # randomly choosing between left,right,center images 
        for i in range(len(df3)):
            view = pos[np.random.choice(3)]
            # print(view)
            if view == 'left':
                yBatch[i] = df3.iloc[i]['steering'] + steerAdjustment
            if view == 'center':
                yBatch[i] = df3.iloc[i]['steering'] + 0
            if view == 'right':
                yBatch[i] = df3.iloc[i]['steering'] - steerAdjustment
            xBatch[i] = plt.imread(loc + str(df3.iloc[i][view]))[crop:,:,:]
        yield (xBatch,yBatch)


def main(_):
    # batch size: read your notes on REAMME.md for what batch size does to training,
    batchSize = FLAGS.batchSize
    pos = {0:'left',1:'center',2:'right'}
    im = plt.imread(loc+ str(df.iloc[0]['center']))
    # seperating data based on steering input. 
    df1 = df[df['steering']!=0]
    df0 = df[df['steering']==0]
    crop = FLAGS.cropSize
    # creating the CNN model via Keras 
    model = Sequential()
    # scaling pixel value so that there are going to be [-0.5,0.5]
    model.add(Lambda(lambda x: x/128 - 0.5,
                     input_shape = (im.shape[0]-crop, im.shape[1], im.shape[2])))
    # scaling down the image size by half 
    model.add(AveragePooling2D((2,2)))
    model.add(Conv2D(5, 5, 5,W_regularizer=l2(0.001)))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2,2)))
    model.add(Conv2D(15, 5, 5,W_regularizer=l2(0.001)))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))
    model.add(Conv2D(20, 5, 5,W_regularizer=l2(0.001)))
    model.add(AveragePooling2D((2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(25, 5, 5,W_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2,2)))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(200, activation='relu',W_regularizer=l2(0.000)))
    model.add((Dropout(0.5)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    model.fit_generator(generate_data(pos,
                                      im,
                                      df1,
                                      df0,
                                      FLAGS.steerAdjustment,
                                      crop,
                                      FLAGS.oversamplingCoef,
                                      batchSize),
                        samples_per_epoch=(len(df) // batchSize) * batchSize,
                        nb_epoch=FLAGS.epoch,
                        verbose=1,
                        validation_data=generate_data(
                                      pos,
                                      im,
                                      dfValidation[dfValidation['steering']!=0],
                                      dfValidation[dfValidation['steering']==0],
                                      FLAGS.steerAdjustment,
                                      crop,
                                      10),
                        nb_val_samples=200)
    # saving the model into memory
    model.save_weights('model.h5')  
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())
    gc.collect()
if __name__ == '__main__':
    tf.app.run()
