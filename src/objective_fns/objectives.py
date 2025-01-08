# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:43:32 2024

@author: siliconsynapse
"""

import jax.numpy as jnp
from jax import jit,vmap

import jax
from jax.tree_util import Partial as partial
def MPC_decorator(cost_to_go,kinematic_model,gamma,state_train=None,thetas=None,method="gymenv"):

    # lower value is better
    if method=="gymenv":
        @jit
        def MPC_obj(U,state):

            states = kinematic_model(U,state)

            horizon,action_dim = U.shape

            # iterate through time step
            Js = [None] * horizon

            for t in range(1,horizon+1):
                # iterate through cost to go for each element!

                J = cost_to_go(state=states[t])
                Js[t-1] = J

            Js = jnp.stack(Js,axis=-1)

            gammas = gamma**(jnp.arange(horizon))

            total_cost = jnp.sum(gammas*Js,axis=-1)/jnp.sum(gammas)

            return total_cost

    elif method=="Single_FIM_3D_evasion_NN_MPPI":
        @jit
        def MPC_obj(U,radar_state,target_state,J,A):

            # horizon = U.shape[1]
            M,dm = target_state.shape
            N,horizon,dn = radar_state.shape

            target_states = kinematic_model(U,target_state,dt)

            # iterate through time step
            Js = [None]*horizon
            for t in range(1,horizon+1):
                # iterate through each FIM corresponding to a target

                J = cost_to_go(radar_state=radar_state[:,t,:3],target_state=target_states[:,t,:3],J=J,method="NN", state_train=state_train)
                Js[t-1] = J

            Js = jnp.stack(Js)
            if N ==1 and M ==1 :
                logdets = jnp.log(Js)
            else :
                _,logdets = jnp.linalg.slogdet(Js)
            gammas = gamma**(jnp.arange(horizon))
            multi_FIM_obj = jnp.sum(gammas*logdets)/jnp.sum(gammas)

            return +multi_FIM_obj


    return MPC_obj