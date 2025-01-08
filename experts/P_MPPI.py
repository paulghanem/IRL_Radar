# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:09:35 2024

@author: siliconsynapse
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import math
from jax import config
config.update("jax_enable_x64", True)

from functools import partial

from src.control.dynamics import cartpole_step,CartPoleEnvState,kinematics
from src.control.MPPI import MPPI_wrapper,weighting,MPPI_scores_wrapper,MPPI_control
from src.objective_fns.cost_to_go_fns import cart_pole_cost
from src.objective_fns.objectives import MPC_decorator

from time import time
import os
import argparse
from copy import deepcopy
import shutil

from tqdm import tqdm

class P_MPPI:
    def __init__(self, state_shape, n_actions):
        

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.MPPI_iterations=10
        self.gamma=0.99
        self.temperature=0.1
        self.num_traj=500
        self.seed = 123
        self.v_init = 0
        self.av_init = 0
        
    def predict_probs(self,mean,cov,x):
       x=x.T
       mean=mean.T
       pdf=(2*math.pi)**(-1)*(jnp.linalg.det(cov))**(-1/2)*jnp.exp(-1/2*jnp.matmul(jnp.matmul((x-mean).T,jnp.linalg.inv(cov)),(x-mean)))
       return pdf
        
    def generate_session(self, args,epochs,state_train,D_demo,mpc_method=None,thetas=None):
        

        states, traj_probs, actions,FIMs = [], [], [],[]
        
        key = jax.random.PRNGKey(args.seed)
        np.random.seed(args.seed)


        # ==================== MPPI CONFIGURATION ================================= #
        dynamic_rollout = jax.jit(partial(kinematics, step_fn=cartpole_step))

        mppi = MPPI_wrapper(dynamic_rollout)

        mpc_obj = MPC_decorator(cart_pole_cost, dynamic_rollout,self.gamma,method="Single_FIM_3D_action_NN_MPPI")

        mppi_scores = MPPI_scores_wrapper(mpc_obj,method="NN")

        weight_fn = weighting()

        action_space = jnp.array([[-10,10]])

        cov_timestep = jnp.eye(args.a_dim)*10
        cov = jax.scipy.linalg.block_diag(*[cov_timestep for _ in range(args.horizon)])


        state = D_demo[0,:args.s_dim]


        U_MPPI_init= jnp.array(np.random.randn(15, args.horizon, args.a_dim) * 0.2)
        U_nominal = U_MPPI_init.mean(axis=0)

        rewards = [cart_pole_cost(state)]

        pbar = tqdm(total=args.N_steps, desc="Starting")

        for step in range(1,args.N_steps+1):
            (U_nominal,cov_prime), (states_nominal, states_MPPI), cost_MPPI, key = MPPI_control(state, U_nominal, cov, key,
                                                                            dynamic_rollout, action_space,
                                                                            mppi, mppi_scores, weight_fn,
                                                                            method="NN",state_train=state_train,
                                                                            args=args)

            state = states_nominal[0]
            rewards.append(cart_pole_cost(state))
            # state_seq_mppi.append(
            #     CartPoleEnvState(t=i + 1, x=state[0], x_dot=state[1], theta=state[2], theta_dot=state[3]))

            # print(U_nominal.shape)
            # print("U: ",U_nominal)

            probs_fun=jax.vmap(self.predict_probs,(0,None,0))
            prob=self.predict_probs(U_nominal[0,:], cov_prime[args.a_dim:(2*args.a_dim),args.a_dim:(2*args.a_dim)],U_nominal[0,:])
            prob=jnp.array([1])


            states.append(state)
            traj_probs.append(prob.flatten())
            actions.append(U_nominal[0,:].flatten())

            pbar.set_description(f"State = {state} , Action = {U_nominal[-1]} , Cost True = {cart_pole_cost(state):.4f} ,  Cost Estimated = {state_train.apply_fn({'params':state_train.params},state.reshape(1,-1)).ravel().item():.4f}")
            pbar.update(1)


        rewards = np.array(rewards)
        pbar.close()
        return states,traj_probs,actions