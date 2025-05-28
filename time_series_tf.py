# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:37:16 2025

@author: siliconsynapse
"""

import matplotlib.pyplot as plt 
from pandas import read_csv
import numpy as np 
import tensorflow as tf 
from sklearn.metrics import mean_squared_error


# load the dataset
dataframe = read_csv('C:/Users/siliconsynapse/Desktop/airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')


def create_dataset(dataset,window_size):
    X,Y=[],[]
    
    for i in range(window_size,len(dataset)):
        X.append(dataset[i-window_size:i])
        Y.append(dataset[i])
        
    
    return np.array(X),np.array(Y)
    

class RNN(tf.keras.Model):
    def __init__(self,hidden_dim_lstm,hidden_dim_MLP,output_dim,window_size,input_dim):
        super().__init__()
        self.lstm1=tf.keras.layers.LSTM(hidden_dim_lstm,input_shape=(window_size,input_dim),activation='tanh',return_sequences=True)
        self.Dense1=tf.keras.layers.Dense(hidden_dim_MLP)
        self.lstm2=tf.keras.layers.LSTM(hidden_dim_lstm,activation='relu')
        self.fc=tf.keras.layers.Dense(output_dim)
        
         
    def call(self,inputs):
        
        x=self.lstm1(inputs)
        x=self.Dense1(x)
        x=self.lstm2(x)
        x=self.fc(x)
        return x 


window_size=10  
X,Y=create_dataset(dataset, window_size)
trainX=X[:int(0.67*len(X)),:,:]
trainY=Y[:int(0.67*len(Y)),:]
testX=X[int(0.67*len(X)):,:,:]
testY=Y[int(0.67*len(Y)):,:]
    
hidden_dim_lstm=64
hidden_dim_MLP=30
input_dim=X.shape[2]
output_dim=Y.shape[1]
  
model=RNN(hidden_dim_lstm,hidden_dim_MLP,output_dim,window_size,input_dim)



model.compile(optimizer='adam',loss="mean_squared_error")
model.fit(trainX,trainY,epochs=100,batch_size=100)

pred=model(testX)
pred=np.array(pred)
rmse=np.sqrt(mean_squared_error(testY,pred))
print(rmse)


    
    
    
        