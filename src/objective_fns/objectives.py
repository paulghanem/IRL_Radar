# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:43:32 2024

@author: siliconsynapse
"""

import jax.numpy as jnp
from jax import jit,vmap

import jax
from jax.tree_util import Partial as partial
def MPC_decorator(cost_to_go,kinematic_model,gamma,state_train=None,thetas=None,method="gymenv",args=None):

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

                J = cost_to_go(state=states[t-1])
                Js[t-1] = J

            Js = jnp.stack(Js,axis=-1)

            gammas = gamma**(jnp.arange(horizon))

            total_cost = jnp.sum(gammas*Js,axis=-1)/jnp.sum(gammas)

            return total_cost

    elif method == "Single_FIM_3D_action_NN_MPPI":
        @jit
        def MPC_obj(U, state,state_train):
            # horizon = U.shape[1]

            states = kinematic_model(U, state)

            horizon, action_dim = U.shape

            # iterate through time step

            Js = [None] * horizon

            for t in range(1, horizon + 1):
                # iterate through each FIM corresponding to a target

                J = state_train.apply_fn({'params':state_train.params},states[t-1].reshape(1,-1)).ravel()

                Js[t - 1] = J

            Js = jnp.stack(Js, axis=-1)

            gammas = gamma**(jnp.arange(horizon))

            total_cost = jnp.sum(gammas*Js,axis=-1)/jnp.sum(gammas)

            if args.gail:
                D = jnp.divide(jnp.exp(-total_cost), (jnp.exp(-total_cost) + 1))
                total_cost = -jnp.log(D)

            return total_cost


    return MPC_obj