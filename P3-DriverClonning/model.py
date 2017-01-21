from __future__ import division
import numpy as np
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
loc = '/media/pemfir/Data/UdacityProjects/Data/DriverClonning/data/'
# command line flags
flags.DEFINE_string('training_file', loc + 'driving_log.csv', "Bottleneck features training file (.p)")
flags.DEFINE_integer('epoch', 15, "number of epoches")
flags.DEFINE_integer('batchSize', 100, "batch size")
flags.DEFINE_integer('cropSize', 60, "number of epoches")
flags.DEFINE_integer('steerAdjustment', 0.25, "number of epoches")
flags.DEFINE_integer('oversamplingCoef', 0.8 , "number of epoches")


def generate_data(df,batchSize,pos,im,start,end,df1,df0,steerAdjustment,crop,oversamplingCoef):
    while(True):
        indices1 = np.random.choice(len(df1), int(oversamplingCoef*batchSize))
        indices2 = np.random.choice(len(df0), int((1-oversamplingCoef)**batchSize))
        df3 = df1.iloc[indices1].append(df0.iloc[indices2])
        xBatch = np.zeros((batchSize, im.shape[0]-crop, im.shape[1], im.shape[2]), dtype=np.float32)
        yBatch = np.zeros((batchSize,), dtype=np.float32)
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
            # print(xBatch[i].shape)
        # start = end
        # end = end + start 
        # if end > len(df3):
        #     start = 0
        #     end = start + batchSize
        yield (xBatch,yBatch)


def main(_):
    df = pd.read_csv(FLAGS.training_file)
    batchSize = FLAGS.batchSize
    pos = {0:'left',1:'center',2:'right'}
    im = plt.imread(loc+ str(df.iloc[0]['center']))
    start = 0 
    end = start + batchSize
    df1 = df[df['steering']!=0]
    df0 = df[df['steering']==0]
    crop = FLAGS.cropSize
    model = Sequential()
    model.add(Lambda(lambda x: x/128 - 0.5,
                     input_shape = (im.shape[0]-crop, im.shape[1], im.shape[2])))
    model.add(AveragePooling2D((2,2)))
    model.add(Conv2D(5, 5, 5,W_regularizer=l2(0.000)))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((2,2)))
    model.add(Conv2D(15, 5, 5,W_regularizer=l2(0.000)))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))
    model.add(Conv2D(20, 5, 5,W_regularizer=l2(0.000)))
    model.add(AveragePooling2D((2,2)))
    model.add(Activation('relu'))
    model.add(Conv2D(25, 5, 5,W_regularizer=l2(0.000)))
    model.add(MaxPooling2D((2,2)))
    model.add((Dropout(0.5)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(200, activation='relu',W_regularizer=l2(0.000)))
    model.add((Dropout(0.5)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    model.fit_generator(generate_data(df,
                                      batchSize,
                                      pos,
                                      im,
                                      start,
                                      end,
                                      df1,
                                      df0,
                                      FLAGS.steerAdjustment,
                                      crop,
                                      FLAGS.oversamplingCoef), samples_per_epoch=(len(df) // batchSize) * batchSize, nb_epoch=15, verbose=1)
    model.save_weights('model.h5')  # always save your weights after training or during training
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())
    gc.collect()
if __name__ == '__main__':
    tf.app.run()