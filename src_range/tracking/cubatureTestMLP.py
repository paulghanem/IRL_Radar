# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:58:16 2024

@author: siliconsynapse
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def place_sensors(xlim,ylim,N):
    N = np.sqrt(N).astype(int)
    xs = np.linspace(xlim[0],xlim[1],N)
    ys = np.linspace(ylim[0],ylim[1],N)
    X,Y = np.meshgrid(xs,ys)
    return np.column_stack((X.ravel(),Y.ravel()))

# Generate data
def generate_data_state(target_state,N, M_target, dm, dt,Q):

    # 2D constant velocity model
    A_single = np.array([[1., 0, 0, dt, 0, 0],
                         [0, 1., 0, 0, dt, 0],
                         [0, 0, 1, 0, 0, dt],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1., 0],
                         [0, 0, 0, 0, 0, 1]])

    A = np.kron(np.eye(M_target), A_single);

    target_state = target_state.reshape(-1,1)

    x = np.zeros((dm*M_target, N)) # state
    noise = np.random.multivariate_normal(np.zeros(dm*M_target,),Q,size=(N,)).T
    for n in range(N):
        target_state = A @ target_state #+ noise[:,[n]]

        x[:, n] =target_state.ravel()

    return x

