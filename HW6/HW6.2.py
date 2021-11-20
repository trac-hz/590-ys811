#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 18:20:33 2021

@author: tracysheng
"""
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow.keras as tk
import matplotlib.pyplot as plt
from keras import models
from keras import layers

import keras
from keras.datasets import mnist,cifar10

#GET DATASET
from keras.datasets import mnist
(X, Y), (test_images, test_labels) = mnist.load_data()

# X=X+np.random.uniform(0,1,28*28)
#NORMALIZE AND RESHAPE
#X=X/np.max(X) 
#X=X.reshape(60000,28*28); 

#MODEL
n_bottleneck=20
N_channels=1
PIX=28
input_img = keras.Input(shape=(PIX, PIX, N_channels))

## ENCODER
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
# # AT THIS POINT THE REPRESENTATION IS (4, 4, 8) I.E. 128-DIMENSIONAL
 
# #DECODER
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)
    

#COMPILE
model = keras.Model(input_img, decoded)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics='accuracy');
model.summary()


model.fit(X, X, epochs=3, batch_size=1000,validation_split=0.2)

MSE_FINAL=0.1 #VARIANCE OF ERROR DISTRIBUTION STD^2
STD_FINAL=np.sqrt(MSE_FINAL)
THRESHOLD=4*STD_FINAL


#PLOT ORIGINAL AND RECONSTRUCTED 
X1=model.predict(X) #DECODED

# summarize history for loss
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# #RESHAPE
# X=X.reshape(60000,28,28); #print(X[0])
# X1=X1.reshape(60000,28,28); #print(X[0])

# #COMPARE ORIGINAL 
# f, ax = plt.subplots(4,1)
# I1=11; I2=46
# ax[0].imshow(X[I1])
# ax[1].imshow(X1[I1])
# ax[2].imshow(X[I2])
# ax[3].imshow(X1[I2])
# plt.show()

#GET MINIST FASHION DATASET
# from keras.datasets import mnist
# (X, Y), (test_images, test_labels) = mnist.load_data()

from keras.datasets import fashion_mnist 
(XF, YF), (test_images, test_labels) = fashion_mnist.load_data()

#reshape XF
XF=XF/np.max(XF) 

#PREDICTION FOR MNIST-FASHION 
XpF=model.predict(XF) 
 
count_anomolies=0; 
for i in range(0,len(XpF)):
    MAE=np.mean(np.absolute(XpF[i]-XF[i]))  #(RECONSTRUCTED-ORIGINIAL)
    if(MAE>THRESHOLD):
        print("found anomoly:")
        count_anomolies+=1

print("total anomolies found=",count_anomolies)
print("percentage anomolies found=",count_anomolies/XpF.shape[0])






