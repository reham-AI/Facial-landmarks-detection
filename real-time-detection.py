# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 21:35:09 2020

@author: Reham
"""

import cv2
from model import create_model

#haarcascade for face detection
faceCascade = cv2.CascadeClassifier('C:\\Users\\Reham\\Anaconda3\\envs\\reko\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
model_weights = 'facialLandmarks.h5'  # Trained model weights
model = create_model()
model.load_weights(model_weights)
input_shape = 96
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if (ret==True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        frame_copy = gray.copy()
        
        for (x, y, w, h) in faces:
            
            #detect the face in the processing frame
            roi = frame_copy[y:y+h, x:x+w] 
            roi = roi / 255.0
            roi_gray_resized = cv2.resize(roi, (input_shape,input_shape))
            roi_gray_resized = roi_gray_resized.reshape(1,input_shape, input_shape, 1)
            #Input the detected face to the model for predicyion      
            pred_image = model.predict(roi_gray_resized)
            pred_image =pred_image*96
            #separate x and y features
            #adjust the features coordinates to the whole frame
            xnew = (pred_image[:,0::2]/ input_shape *w ) + x
            ynew = (pred_image[:,1::2]/ input_shape *h ) + y
    
            for i in range (xnew.shape[1]):   # loop for the 15 landmarks
                cv2.circle(frame,(xnew[:,i],ynew[:,i]),1,(0,0,255))   #(0,0,255) for red color
                   
        # Display the resulting frame
        cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()