# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:37:16 2025

@author: siliconsynapse
"""

import matplotlib.pyplot as plt 
from pandas import read_csv
import numpy as np 
import torch
from torch import nn, optim 
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
    

class RNN(nn.Module):
    def __init__(self,hidden_dim_lstm,hidden_dim_MLP,output_dim,window_size,input_dim):
        super().__init__()
        self.lstm1=nn.LSTM(hidden_size=hidden_dim_lstm,input_size=input_dim,batch_first=True)
        self.relu1=nn.ReLU()
        self.Dense1=nn.Linear(in_features=hidden_dim_lstm,out_features=hidden_dim_MLP)
        self.lstm2=nn.LSTM(hidden_size=hidden_dim_lstm,input_size=hidden_dim_MLP)
        self.relu2=nn.ReLU()
        self.fc=nn.Linear(in_features=hidden_dim_lstm,out_features=output_dim)
        
         
    def forward(self,inputs):
        
        x,_=self.lstm1(inputs)
        x=self.Dense1(x)
        x,_=self.lstm2(x)
        x=self.fc(x)
        return x[:,-1,:]


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
loss_fn=nn.MSELoss()
epochs=100
batch_size=10

model=RNN(hidden_dim_lstm,hidden_dim_MLP,output_dim,window_size,input_dim)
optimizer=optim.Adam(model.parameters(),lr=0.001)
for epoch in range (epochs):
    for i in range(batch_size,len(trainX)):
        x=trainX[i-batch_size:i,:,:]
        y=trainY[i-batch_size:i,:]
        y_pred=model(torch.tensor(x))
        loss=loss_fn(torch.tensor(y),y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        
        


pred=model(torch.tensor(testX))
pred=np.array(pred.detach())
rmse=np.sqrt(mean_squared_error(testY,pred))
print(rmse)