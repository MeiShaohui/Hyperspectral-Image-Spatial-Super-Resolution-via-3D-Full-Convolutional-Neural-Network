from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Convolution3D

import  keras
import h5py
from keras.optimizers import Adam

def srcnn(input_shape=(33,33,110,1)):
    #for ROSIS  sensor
    model = Sequential()
    model.add(Convolution3D(64, 9, 9, 7, input_shape=input_shape, activation='relu'))
    model.add(Convolution3D(32, 1, 1, 1, activation='relu'))
    model.add(Convolution3D(9, 1, 1, 1, activation='relu'))
    model.add(Convolution3D(1, 5, 5, 3))
    model.compile(Adam(lr=0.00005), 'mse')
    return model




