from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import jit

from flax import struct
import chex
from typing import Any, Dict, Optional, Tuple, Union
from jax import lax
from gymnax.environments import environment
from gymnax.environments import spaces

@struct.dataclass
class CartPoleEnvState(environment.EnvState):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int

@struct.dataclass
class PendulumEnvState(environment.EnvState):
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    last_u: jnp.ndarray  # Only needed for rendering
    time: int

@struct.dataclass
class MountainCar(environment.EnvState):
    position: jnp.ndarray
    velocity: jnp.ndarray
    time: int

def cartpole_step(
        action,
        state):

    gravity = 9.8
    masscart = 1.0
    masspole= 0.1
    total_mass = masscart + masspole  # (masscart + masspole)
    length = 0.5
    polemass_length = masspole * length  # (masspole * length)
    force_mag = 10.0
    tau = 0.02

    x, x_dot, theta, theta_dot = state[...,0],state[...,1],state[...,2],state[...,3]

    force = action[...,0]

    """Performs step transitions in the environment."""
    force = jnp.clip(force,-force_mag,force_mag) #force_mag * action - force_mag * (1 - action) turn to continuous :)
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)

    temp = (
                   force + polemass_length * theta_dot ** 2 * sintheta
           ) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
            length
            * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    # Only default Euler integration option available here!
    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc


    return lax.stop_gradient(jnp.array([x, x_dot, theta, theta_dot]))



def pendulum_step(
    action,state
):
    max_speed = 8.0
    max_torque = 2.0
    dt =  0.05
    g = 10.0  # gravity
    m = 1.0  # mass
    l = 1.0  # length

    """Integrate pendulum ODE and return transition."""
    u = jnp.clip(action, -max_torque, max_torque)

    theta,theta_dot = state[...,0],state[...,1]

    newthdot = theta_dot + (
        (
            3 * g / (2 * l) * jnp.sin(theta)
            + 3.0 / (m * l**2) * u
        )
        * dt
    )

    newthdot = jnp.clip(newthdot, -max_speed, max_speed)
    newth = theta + newthdot * dt


    theta = newth.squeeze()
    theta_dot = newthdot.squeeze()

    return lax.stop_gradient(jnp.array([theta, theta_dot]))



def get_action_space(env_name):
    if env_name == "CartPole-v1":
        return jnp.array([[-10.,10.]])

    if env_name == "Pendulum-v1":
        return jnp.array([[-2.,2.]])

def get_state(state,action=None,time=None,env_name="CartPole-v1"):
    if env_name == "CartPole-v1":
        return CartPoleEnvState(x=state[0],x_dot=state[1],theta=state[2],theta_dot=state[3],time=time)

    if env_name == "Pendulum-v1":
        return PendulumEnvState(theta=state[0],theta_dot=state[1],last_u=action,time=time)


def get_step_model(env_name):
    if env_name == "CartPole-v1":
        return cartpole_step
    if env_name == "Pendulum-v1":
        return pendulum_step


def kinematics(action, state, step_fn):
    """Rollout a jitted gymnax episode with lax.scan."""

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""

        action = tmp
        next_state = step_fn(action,state_input)

        carry = next_state
        return carry, carry

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        policy_step,
        state,
        action,
    )
    # Return masked sum of rewards accumulated by agent in episode
    states = scan_out
    return states

