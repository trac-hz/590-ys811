#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:59:33 2021

@author: tracysheng
"""

#Create a labeled dataset by selecting and downloading at least 3 novels 


import os
imdb_dir = '/Users/tracysheng/Desktop/590/HW5'
train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []
for label_type in ['BattleOfAusterlitz', 'BylinyBook','Newshound']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'BattleOfAusterlitz':
                labels.append(1)
            elif label_type == 'BylinyBook':
                labels.append(2)
            else :
                labels.append(3)
                
print(labels)
print(len(texts))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100 #Cuts off reviews after 100 words 
training_samples = 130 #Trains on 100 samples
validation_samples = 32
#Validates on 10,000 samples 
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
#Considers only the top 10,000 words in the dataset

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
#Splits the data into a training set and a validation set, but first shuffles the data, 
#because you’re starting with data in which samples are ordered (all negative first, then all positive)
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples] 
y_val = labels[training_samples: training_samples + validation_samples]

import numpy
a = numpy.asarray(x_train)
numpy.savetxt("x_train.csv", a, delimiter=",")
b = numpy.asarray(x_train)
numpy.savetxt("y_train.csv", b, delimiter=",")
c = numpy.asarray(x_val )
numpy.savetxt("x_val.csv", c, delimiter=",")
d = numpy.asarray(y_val)
numpy.savetxt("y_val.csv", d, delimiter=",")



