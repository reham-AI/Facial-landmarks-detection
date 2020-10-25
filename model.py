# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 21:32:39 2020

@author: Reham
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import save_model, load_model


def create_model():
    # CNN Model architecture
    model = Sequential()
    # Input Layer
    # The convolution layer
    model.add(Conv2D(32, (3, 3),
                     padding = 'same',
                     activation='tanh',
                     input_shape=(96, 96,1)))
    
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    
    model.add(Dense(256, activation='tanh'))
    
    model.add(Dropout(0.5))
    
    model.add(Dense(30, activation='sigmoid'))
        
    print(model.summary())
       
    return model
