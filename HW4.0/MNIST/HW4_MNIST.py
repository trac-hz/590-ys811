#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:51:04 2021

@author: tracysheng
"""
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras import models
from keras import layers 
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model

dataset = 'fashion_mnist'

if (dataset == 'mnist'):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
elif (dataset == 'cifar10'):
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
else:
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
#Add function to visualize a random (or specified) image in the dataset

def show_image(x):
    
    image=train_images[x]
        
    image = resize(image, (10, 10), anti_aliasing=True)
    plt.imshow(image, cmap=plt.cm.gray); plt.show()
#################################################################
    
##train CNN model
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

#NORMALIZE
train_images = train_images.astype('float32') / 255 
test_images = test_images.astype('float32') / 255  

#DEBUGGING
NKEEP=10000
batch_size=int(0.05*NKEEP)
epochs=20
print("batch_size",batch_size)
rand_indices = np.random.permutation(train_images.shape[0])
train_images=train_images[rand_indices[0:NKEEP],:,:]
train_labels=train_labels[rand_indices[0:NKEEP]]


#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
tmp=train_labels[0]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(tmp, '-->',train_labels[0])
print("train_labels shape:", train_labels.shape)

X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

#BUILD MODEL SEQUENTIALLY (LINEAR STACK)
#-------------------------------------
model = models.Sequential()
if (dataset == 'cifar10'):
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
else:
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()
#-------------------------------------
#COMPILE AND TRAIN MODEL
#-------------------------------------
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)


#-------------------------------------
#EVALUATE ON TEST DATA
#-------------------------------------
train_loss, train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size)
test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=test_images.shape[0])
print('train_acc:', train_acc)
print('test_acc:', test_acc)

##data-augmentation
datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

i=0
x= train_images[1,:,:]
x=x.reshape((1,) + x.shape)

for batch in datagen.flow(x, batch_size=1):
    
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()


#BASIC PLOTTING 
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(X_val, y_val))
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

##save model
model.save('CNN_Model')

##load model
model = load_model('CNN_Model')
model.summary()  

##Visualizing intermediate activations
img = train_images[1,:,:]
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0) 
img_tensor /= 255.
print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()

layer_outputs = [layer.output for layer in model.layers[:8]] 
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')

layer_names = []
for layer in model.layers[:5]:
    layer_names.append(layer.name)
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                         :, :,
                                  col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                     row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                    scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')





















