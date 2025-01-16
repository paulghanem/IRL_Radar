import jax.numpy as jnp
from jax import jit,vmap

import jax
from jax.tree_util import Partial as partial
import jax.numpy as jnp
import numpy as np

def cart_pole_cost(state,goal_x = 0.0,goal_theta = 0.0):
    x = state[...,0]
    theta = state[...,2]

    pos_cost = (x-goal_x)**2
    theta_cost = (theta-goal_theta)**2

    return 5*pos_cost + 5*theta_cost

def pendudulum_cost(state):
    x,y,theta_vel = state[...,0],state[...,1],state[...,2]

    # see gymlibrary dev https://www.gymlibrary.dev/environments/classic_control/pendulum/
    return ((x-1)**2 + y**2 + 0.1 * theta_vel**2)


def mountaincar_cost(state):
    position = state[...,0]
    velocity = state[...,1]
    goal_position = 0.45
    goal_velocity = 0.0

    # cost = 100 * (
    #     np.logical_or(position <= goal_position,velocity <= goal_velocity)
    # )
    cost = 10 * (position - goal_position)**2
    return cost



def get_cost(env_name):
    if env_name == "CartPole-v1":
        return cart_pole_cost
    if env_name == "Pendulum-v1":
        return pendudulum_cost
    if env_name == "MountainCarContinuous-v0":
        return mountaincar_cost
