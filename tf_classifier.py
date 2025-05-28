# -*- coding: utf-8 -*-
"""
Created on Mon May 26 23:54:25 2025

@author: siliconsynapse
"""

import tensorflow as tf 

import numpy as np 

from sklearn.metrics import mean_squared_error


    





class CNN(tf.keras.Model):
    def __init__(self,):
        super().__init__()
        self.cnn1=tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3))
        self.dense1=tf.keras.layers.Dense(64)
        self.maxpool1=tf.keras.layers.MaxPooling2D((2,2))
        self.cnn2=tf.keras.layers.Conv2D(64,(3,3),activation='relu')
        self.cnn3=tf.keras.layers.Conv2D(64,(3,3),activation='relu')
        self.maxpool2=tf.keras.layers.MaxPooling2D((2,2))
        self.dense2=tf.keras.layers.Dense(64,activation='relu')
        self.flatten=tf.keras.layers.Flatten()
        self.fc=tf.keras.layers.Dense(10)
        
        
    def call(self,inputs):
        x=self.cnn1(inputs)
        x=self.cnn2(x)
        x=self.cnn3(x)
        x=self.flatten(x)
        x=self.dense2(x)
        x=self.fc(x)
        return x 


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()    

x_train=x_train / 255.0

x_test=x_test / 255.0

normalizer=tf.keras.layers.Normalization()
normalizer.adapt(x_train)
x_train=normalizer.normalize(x_train)
x_test=normalizer.normalize(x_test)

batch_size=100
epochs=100   
model=CNN()
model.compile(optimizer="Adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs)
y_pred=model(x_test)
rmse=np.sqrt(mean_squared_error(y_pred,y_test))
