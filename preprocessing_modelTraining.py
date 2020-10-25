
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:22:46 2020

@author: Reham
"""

import numpy as np
import pandas as pd

#pictures
face_images_db = np.moveaxis(np.load('face_images.npz')['face_images'], -1, 0)
#30 features of 15 different landmarks
facial_keypoints_df = pd.read_csv('facial_keypoints.csv')

# keep only data with 15 key points 
rows_with_nan = []
for index, row in facial_keypoints_df.iterrows():
    is_nan_series = row.isnull()
    if is_nan_series.any():
        rows_with_nan.append(index)
faceImagesDB=np.delete(face_images_db ,rows_with_nan ,0 )
facialKeypointsDF = facial_keypoints_df.dropna(how='any',axis=0)


# prepare images and features to be input to the model 

faceImagesDB = faceImagesDB.reshape (faceImagesDB.shape[0],faceImagesDB.shape[1],faceImagesDB.shape[2],1) # (2140,96,96,1)
faceImagesDB = faceImagesDB / 255.0 # normalize images
facialKeypointsDF = facialKeypointsDF / faceImagesDB.shape[1]  # normalyize features to be between 0-1

# # Split the dataset to train and test
from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(faceImagesDB, facialKeypointsDF, test_size=0.2)
Xtrain = np.asarray(Xtrain)
Xtest = np.asarray(Xtest)
Ytrain = np.asarray(Ytrain)
Ytest = np.asarray(Ytest)
print ('xtest', Xtest.shape)
print ('Ytest', Ytest.shape)

# train the model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD

model = Sequential()
model.add(Conv2D(32, (3, 3), padding = 'same', activation='tanh', input_shape=(96, 96,1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd , metrics=['accuracy'])
model.fit(Xtrain, Ytrain, batch_size=128, epochs=20, validation_data = (Xtest, Ytest), verbose = 1)

#save the model
model.save('facialLandmarks.h5')
results = model.evaluate(Xtest, Ytest, batch_size=128)
print("test loss, test acc:", results)










