# %load feature_extractionv2.py
import pickle
import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.utils import np_utils

from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation
from keras.datasets import cifar10
# (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train = X_train / 255 - 0.5
# X_test = X_test / 255 - 0.5

# TODO: Re-construct the network and add dropout after the pooling layer.
from keras.layers import Dropout
from keras.regularizers import l2, activity_l2

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epoch', 10, "number of epoches")
flags.DEFINE_integer('batchSize', 128, "batch size")
def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    
    print("Training file", training_file)
    print("Validation file", validation_file)
    
    
    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']
    
    classSize = len(np.unique(y_train))
    print("class size", classSize)
    
    y_train = np_utils.to_categorical(y_train, classSize)
    y_val = np_utils.to_categorical(y_val, classSize)
    return X_train, y_train, X_val, y_val,classSize


def main(_):
    # load bottleneck data

    X_train, y_train, X_val, y_val,classSize = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    model = Sequential()
    model.add(Flatten(input_shape = X_train.shape[1:]))
    model.add(Dense(128, activation='relu',W_regularizer=l2(0.001)))
    # model.add((Dropout(0.5)))
    model.add(Dense(classSize, activation='softmax',W_regularizer=l2(0.001)))
    model.summary()
# TODO: Compile and train the model here.
    opt = keras.optimizers.Nadam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train,
                        batch_size=FLAGS.batchSize, nb_epoch=FLAGS.epoch,
                        verbose=2, validation_data=(X_val, y_val), shuffle=True)
    # TODO: train your model here


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
