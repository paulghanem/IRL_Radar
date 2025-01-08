import jax.numpy as jnp
from jax import jit,vmap

import jax
from jax.tree_util import Partial as partial
import jax.numpy as jnp

def cart_pole_cost(state,goal_x = 0.0,goal_theta = 0.0):
    x = state[...,0]
    theta = state[...,2]

    pos_cost = (x-goal_x)**2
    theta_cost = (theta-goal_theta)**2

    return 2.5*pos_cost + 5*theta_cost

