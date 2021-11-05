#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:46:19 2021

@author: tracysheng
"""
import pandas as pd
x_train=pd.read_csv('x_train.csv')
y_train=pd.read_csv('y_train.csv')
x_test=pd.read_csv('x_val.csv')
y_test=pd.read_csv('y_val.csv')

#Train both a 1D CNN and a RNN model to predict the novel title (category) 
#Training and evaluating a simple 1D convnet 

from keras.models import Sequential
from keras import layers
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Embedding, SimpleRNN
from sklearn.model_selection import train_test_split

max_features = 10000
max_len = 500
NKEEP=10000
batch_size=int(0.05*NKEEP)
epochs=20
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train,
                                                  test_size=0.2, random_state=42)

model='CNN'
#X_train1=X_train.to_numpy()
#X_val1=X_val.to_numpy()
#Y_train1=Y_train.to_numpy()
#Y_val1=Y_val.to_numpy()
#X_train1.reshape(X_train1.shape[0],X_train1.shape[1],1).astype('float32')
#X_val1.reshape(X_val1.shape[0],X_val1.shape[1],1).astype('float32')
#Y_train1.reshape(Y_train1.shape[0],Y_train1.shape[1],1).astype('float32')
#Y_val1.reshape(Y_val1.shape[0],Y_val1.shape[1],1).astype('float32')

if model=='CNN':
    model = Sequential()
    model.add(layers.Embedding(max_features, 128, input_length=max_len))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))
    model.summary()

elif model=='RNN':

    ##Training and evaluating a RNN model    
    model = Sequential()
    model.add(layers.Embedding(10000, 32))
    model.add(layers.SimpleRNN(32, return_sequences=True))
    model.add(layers.SimpleRNN(32, return_sequences=True))
    model.add(layers.SimpleRNN(32, return_sequences=True))
    model.add(layers.SimpleRNN(32))
    model.summary()

#COMPILE AND TRAIN MODEL
#-------------------------------------
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])  
#-------------------------------------

#EVALUATE ON TEST DATA
#-------------------------------------
train_loss, train_acc = model.evaluate(X_train,Y_train, batch_size=batch_size)
test_loss, test_acc = model.evaluate(x_test,y_test, batch_size=x_test.shape[0])
print('train_acc:', train_acc)
print('test_acc:', test_acc)


#BASIC PLOTTING 
import matplotlib.pyplot as plt

history = model.fit(X_train,
                    Y_train,
                    epochs=20,
                    batch_size=512,
                    validation_split=0.2)

history_dict = history.history
history_dict.keys()


## Plot the training and validation loss
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


## Plot the training and validation accuracy
plt.clf() 
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
