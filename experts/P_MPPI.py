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

from src.control.dynamics import cartpole_step,get_step_model,get_action_space,kinematics
from src.control.MPPI import MPPI_wrapper,weighting,MPPI_scores_wrapper,MPPI_control
from src.objective_fns.cost_to_go_fns import get_cost
from src.objective_fns.objectives import MPC_decorator

from cost_jax import get_gradients,get_hessian

from time import time
import os
import argparse
from copy import deepcopy
import shutil

from tqdm import tqdm

class P_MPPI:
    def __init__(self, state_shape, n_actions,args):
        

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.MPPI_iterations=10
        self.gamma=args.gamma
        self.temperature=0.1
        self.num_traj=500
        self.seed = 123
        self.args = args

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
        dynamic_rollout = jax.jit(partial(kinematics, step_fn=get_step_model(args.gym_env)))

        mppi = MPPI_wrapper(dynamic_rollout)

        mpc_obj = MPC_decorator(None, dynamic_rollout,self.gamma,method="Single_FIM_3D_action_NN_MPPI",args=args)

        mppi_scores = MPPI_scores_wrapper(mpc_obj,method="NN")

        weight_fn = weighting()

        action_space = get_action_space(args.gym_env)

        def cov_coef(env_name):
            if env_name == "CartPole-v1":
                return 10.
            if env_name == "Pendulum-v1":
                return 1.0
            if env_name == "MountainCarContinuous-v0":
                return 0.7

        cov_timestep = jnp.eye(args.a_dim)*cov_coef(args.gym_env)
        cov = jax.scipy.linalg.block_diag(*[cov_timestep for _ in range(args.horizon)])


        state = D_demo[0,:args.s_dim]

        U_MPPI_init= jnp.array(np.random.randn(15, args.horizon, args.a_dim) * 0.2)
        U_nominal = U_MPPI_init.mean(axis=0)
        true_cost_fn = get_cost(args.gym_env)

        rewards = [true_cost_fn(state)]

        pbar = tqdm(total=args.N_steps, desc="Starting")

        for step in range(1,args.N_steps+1):
            states.append(state)

            (U_nominal,cov_prime), (states_nominal, states_MPPI), cost_MPPI, key = MPPI_control(state, U_nominal, cov, key,
                                                                            dynamic_rollout, action_space,
                                                                            mppi, mppi_scores, weight_fn,
                                                                            method="NN",state_train=state_train,
                                                                            args=args)

            state = states_nominal[0]
            rewards.append(true_cost_fn(state))
            # state_seq_mppi.append(
            #     CartPoleEnvState(t=i + 1, x=state[0], x_dot=state[1], theta=state[2], theta_dot=state[3]))

            # print(U_nominal.shape)
            # print("U: ",U_nominal)

            probs_fun=jax.vmap(self.predict_probs,(0,None,0))
            prob=self.predict_probs(U_nominal[-1,:], cov_prime[args.a_dim:(2*args.a_dim),args.a_dim:(2*args.a_dim)],U_nominal[-1,:])
            prob=jnp.array([1])


            traj_probs.append(prob.flatten())
            actions.append(U_nominal[-1,:].flatten())

            pbar.set_description(f"State = {state} , Action = {U_nominal[-1]} , Cost True = {true_cost_fn(state):.4f} ,  Cost Estimated = {state_train.apply_fn({'params':state_train.params},state.reshape(1,-1)).ravel().item():.4f}")
            pbar.update(1)


        rewards = np.array(rewards)
        pbar.close()
        return states,traj_probs,actions
    

    def RGCL(self, args,params,epochs,state_train,D_demo,mpc_method=None,thetas=None):
         
         theta=jnp.concatenate((params['Dense_0']['bias'].flatten(),params['Dense_0']['kernel'].flatten(),params['Dense_1']['bias'].flatten(),params['Dense_1']['kernel'].flatten()))
         n_theta=len(theta)
         P_theta=1e-1*jnp.identity(n_theta)
         
         Q_theta=1e-3*jnp.identity(n_theta)
         states, traj_probs, actions,FIMs = [], [], [],[]
         
         key = jax.random.PRNGKey(args.seed)
         np.random.seed(args.seed)
    
    
         # ==================== MPPI CONFIGURATION ================================= #
         dynamic_rollout = jax.jit(partial(kinematics, step_fn=get_step_model(args.gym_env)))

         mppi = MPPI_wrapper(dynamic_rollout)
    
         mpc_obj = MPC_decorator(None, dynamic_rollout,self.gamma,method="Single_FIM_3D_action_NN_MPPI",args=args)
    
         mppi_scores = MPPI_scores_wrapper(mpc_obj,method="NN")
    
         weight_fn = weighting()

         action_space = get_action_space(args.gym_env)

         def cov_coef(env_name):
             if env_name == "CartPole-v1":
                 return 10.
             if env_name == "Pendulum-v1":
                 return 1.5
             if env_name == "MountainCarContinuous-v0":
                 return 0.7

         cov_timestep = jnp.eye(args.a_dim) * cov_coef(args.gym_env)
         cov = jax.scipy.linalg.block_diag(*[cov_timestep for _ in range(args.horizon)])
    
    
         state = D_demo[0,:args.s_dim]
    
    
         U_MPPI_init= jnp.array(np.random.randn(15, args.horizon, args.a_dim) * 0.2)
         U_nominal = U_MPPI_init.mean(axis=0)
         true_cost_fn = get_cost(args.gym_env)

         rewards = [true_cost_fn(state)]
    
         pbar = tqdm(total=args.N_steps, desc="Starting")
         
         states_expert,actions_expert = D_demo[:,:state.shape[0]], D_demo[:,state.shape[0]:]
   
         for step in range(1,args.N_steps+1):
             (U_nominal,cov_prime), (states_nominal, states_MPPI), cost_MPPI, key = MPPI_control(state, U_nominal, cov, key,
                                                                             dynamic_rollout, action_space,
                                                                             mppi, mppi_scores, weight_fn,
                                                                             method="NN",state_train=state_train,
                                                                             args=args)
    
             state = states_nominal[0]
             rewards.append(true_cost_fn(state))
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
    
             pbar.set_description(f"State = {state} , Action = {U_nominal[-1]} , Cost True = {true_cost_fn(state):.4f} ,  Cost Estimated = {state_train.apply_fn({'params':state_train.params},state.reshape(1,-1)).ravel().item():.4f}")
             pbar.update(1)
             
             gradient_s=get_gradients(state_train,params,state,args.N_steps)
             gradient_d=get_gradients(state_train,params,states_expert[step-1],args.N_steps)
             hessian_s=get_hessian(state_train,params,state,args.N_steps)
             hessian_d=get_hessian(state_train,params,states_expert[step-1],args.N_steps)
            
    
             P_theta=jnp.linalg.inv(jnp.linalg.inv(P_theta +Q_theta) + hessian_d - hessian_s)
            #print(gradient_d)
            #print(gradient_s)
           
             theta=theta-jnp.matmul(P_theta,gradient_d-gradient_s)
            
             params['Dense_0']['bias']=theta[:len(params['Dense_0']['bias'])]
             params['Dense_0']['kernel']=theta[len(params['Dense_0']['bias']):len(params['Dense_0']['bias'])+params['Dense_0']['kernel'].shape[0]*params['Dense_0']['kernel'].shape[1]].reshape(params['Dense_0']['kernel'].shape)
             params['Dense_1']['bias']=theta[len(params['Dense_0']['bias'])+params['Dense_0']['kernel'].shape[0]*params['Dense_0']['kernel'].shape[1]:len(params['Dense_0']['bias'])+params['Dense_0']['kernel'].shape[0]*params['Dense_0']['kernel'].shape[1]+len(params['Dense_1']['bias'])]
             params['Dense_1']['kernel']=theta[len(params['Dense_0']['bias'])+params['Dense_0']['kernel'].shape[0]*params['Dense_0']['kernel'].shape[1]+len(params['Dense_1']['bias']):].reshape(-1,1)
            
        
    
         rewards = np.array(rewards)
         pbar.close()
         return states,traj_probs,actions