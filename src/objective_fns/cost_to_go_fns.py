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

def pendudulum_cost(state):
    theta,theta_vel = state[0],state[1]
    theta_normalize = ((theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    # see gymlibrary dev https://www.gymlibrary.dev/environments/classic_control/pendulum/
    return (theta_normalize**2 + 0.1 * theta_vel**2)



def get_cost(env_name):
    if env_name == "CartPole-v1":
        return cart_pole_cost
    if env_name == "Pendulum-v1":
        return pendudulum_cost
