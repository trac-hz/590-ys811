# HW6.1
#use Keras to train a deep feed forward auto-encoder on the MNIST dataset

#Then use this trained AE to perform anomaly detection using the MNIST-FASHION data 

#Compare the loss of mnist dataset and mnist-fashion dataset
#######################################################################################


##theory: Autoencoders tries to minimize the reconstruction error as part of its training. 
## Anomalies are detected by checking the magnitude of the reconstruction loss.

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow.keras as tk
import matplotlib.pyplot as plt
from keras import models
from keras import layers

#GET DATASET
from keras.datasets import mnist
(X, Y), (test_images, test_labels) = mnist.load_data()

#NORMALIZE AND RESHAPE
X=X/np.max(X) 
X=X.reshape(60000,28*28); 

#MODEL
n_bottleneck=20

# SHALLOW
model = models.Sequential()
model.add(layers.Dense(n_bottleneck, activation='linear', input_shape=(28 * 28,)))
model.add(layers.Dense(28*28,  activation='linear'))
model.add(layers.Dense(28*28,  activation='linear'))

#COMPILE AND FIT
model.compile(optimizer='rmsprop',
               loss='mean_squared_error')
model.summary()
model.fit(X, X, epochs=10, batch_size=1000,validation_split=0.2)

#EXTRACT MIDDLE LAYER (REDUCED REPRESENTATION)
from keras import Model 
extract = Model(model.inputs, model.layers[-2].output) # Dense(128,...)
X1 = extract.predict(X)
print(X1.shape)

#2D PLOT
plt.scatter(X1[:,0], X1[:,1], c=Y, cmap='tab10')
plt.show()

#3D PLOT
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=X1[:,0], 
    ys=X1[:,1], 
    zs=X1[:,2], 
    c=Y, 
    cmap='tab10'
)
plt.show()

#PLOT ORIGINAL AND RECONSTRUCTED 
X1=model.predict(X) 

#RESHAPE
X=X.reshape(60000,28,28); #print(X[0])
X1=X1.reshape(60000,28,28); #print(X[0])

#COMPARE ORIGINAL 
f, ax = plt.subplots(4,1)
I1=11; I2=46
ax[0].imshow(X[I1])
ax[1].imshow(X1[I1])
ax[2].imshow(X[I2])
ax[3].imshow(X1[I2])
plt.show()

#GET MINIST FASHION DATASET
# from keras.datasets import mnist
# (X, Y), (test_images, test_labels) = mnist.load_data()
from keras.datasets import fashion_mnist 
(X, Y), (test_images, test_labels) = fashion_mnist.load_data()

#define the error threshold
threshold=4*Model.evaluate(X,X,batch_size=X.shape[0])
if(err>threshold)-->anomaly






