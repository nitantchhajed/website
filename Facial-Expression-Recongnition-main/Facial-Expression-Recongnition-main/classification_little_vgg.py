# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 20:42:30 2020

@author: harshaj
"""

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator #from image file calling this class to generate image data
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D  
import os

num_classes = 5 #Number of expression we will train
img_rows,img_cols = 48,48 #Size of the image
batch_size = 32 #Number of image we will take at a time.

train_data_dir = '/Python Codes/Facial Expression Recognisation Project/face-expression-recognition-dataset/images/train'
validation_data_dir = '/Python Codes/Facial Expression Recognisation Project/face-expression-recognition-dataset/images/validation'

train_datagen = ImageDataGenerator(
					rescale=1./255, #rescaling image
					rotation_range=30, #rotating image
					shear_range=0.3,
					zoom_range=0.3, #zooming image
					width_shift_range=0.4, #shifting by width
					height_shift_range=0.4, #shifting by height
					horizontal_flip=True, #flipping or mirroring the image
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255) #validating or cross checking data

train_generator = train_datagen.flow_from_directory( 
					train_data_dir,
					color_mode='grayscale',
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical', #Category wise data divided
							shuffle=True)


model = Sequential() #model type is sequential model

# Block-1

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1))) #creating 2-D convolution layer as a part of our cnn architecture
model.add(Activation('elu')) #setting a threshold value and if its cross a minimum value then it will go for further layer.
model.add(BatchNormalization()) #this is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch. basically to stablilize the process
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) #this will eliminate less important features and take only maximum element from the feature map
model.add(Dropout(0.2)) #randomly selected neurons are ignored during training

# Block-2 

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal')) #he_normal=It draws samples from a truncated normal distribution centered on 0.
model.add(Activation('elu')) # function that tend to converge cost to zero faster and produce more accurate results
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal')) #padding=how many pixel it want to take once.
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-4 

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-5

model.add(Flatten()) #converting data into 1-D array.
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6

model.add(Dense(64,kernel_initializer='he_normal')) #the regular deeply connected neural network layer
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5)) 

# Block-7

model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax')) #softmax used for more than 2 samples and sigmoid for binary, here to have the output for those 5 samples

print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint('Emotion_little_vgg.h5', #generating the file, from 2nd epoches it will overwrite saving the model with best accuracy
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss', #if our accuracy is not improving close the training.
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss', #if model accuracy is not improving then reduce the learning rate.
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy', #compiling the model
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 24176
nb_validation_samples = 3006
epochs=25

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)
