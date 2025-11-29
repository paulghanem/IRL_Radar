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
import pdb
from mujoco import mjx 
from functools import partial


@struct.dataclass
class CartPoleEnvState(environment.EnvState):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray
    time: int

@struct.dataclass
class PendulumEnvState(environment.EnvState):
    x: jnp.ndarray
    y: jnp.ndarray
    theta_dot: jnp.ndarray
    last_u: jnp.ndarray  # Only needed for rendering
    time: int

@struct.dataclass
class MountainCar(environment.EnvState):
    position: jnp.ndarray
    velocity: jnp.ndarray
    time: int


@partial(jax.jit, static_argnames=("gym_env"))
def mjx_step(mjx_model, mjx_data, state,action,gym_env):
    """
    JAX-compatible MJX step function.

    Args:
        mjx_model (mjx.Model): The MJX-compiled MuJoCo model.
        mjx_data (mjx.Data): Current simulation data (state).
        action (jnp.ndarray): Control input to apply.

    Returns:
        mjx.Data: Updated simulation state.
    """

    mjx_data = mjx_data.replace(qpos=state[:mjx_model.nq],qvel=state[mjx_model.nq:],ctrl=action)

    # Advance simulation
    mjx_data = mjx.step(mjx_model, mjx_data)
    res=jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
    
    return res


def mountaincar_step(
    state,action
):
    min_action = -1.0
    max_action = 1.0
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.45 # seems a bit odd???
    goal_velocity = 0.0
    power = 0.0015
    gravity = 0.0025
    # max_steps_in_episode: int = 999

    position = state[...,0].reshape(-1, 1)
    velocity = state[...,1].reshape(-1, 1)

    """Perform single timestep state transition."""
    force = jnp.clip(action[...,0].reshape(-1, 1), min_action, max_action)
    velocity = (
        velocity
        + force * power
        - jnp.cos(3 * position) * gravity
    )
    velocity = jnp.clip(velocity, -max_speed, max_speed)
    position = position + velocity
    position = jnp.clip(position, min_position, max_position)

    # this seems like cheating somehow - but it is in the original code
    velocity = velocity * (1 - (position >= goal_position) * (velocity < 0))


    # Update state dict and evaluate termination conditions

    return jnp.concatenate([position, velocity],axis=-1)


def cartpole_step(
        state,
        action):

    gravity = 9.8
    masscart = 1.0
    masspole= 0.1
    total_mass = masscart + masspole  # (masscart + masspole)
    length = 0.5
    polemass_length = masspole * length  # (masspole * length)
    force_mag = 10.0
    tau = 0.02

    x, x_dot, theta, theta_dot = state[...,0].reshape(-1, 1),state[...,1].reshape(-1, 1),state[...,2].reshape(-1, 1),state[...,3].reshape(-1, 1)

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

    return jnp.concatenate([x, x_dot,theta,theta_dot], axis=-1)

def pendulum_step(
    state,action
):
    max_speed = 8.0
    max_torque = 2.0
    dt =  0.05
    g = 10.0  # gravity
    m = 1.0  # mass
    l = 1.0  # length

    """Integrate pendulum ODE and return transition."""
    u = jnp.clip(action[...,0], -max_torque, max_torque)

    x,y,theta_dot = state[...,0].reshape(-1, 1),state[...,1].reshape(-1, 1),state[...,2].reshape(-1, 1)

    theta = jnp.atan2(y,x)

    newthdot = theta_dot + (
        (
            3 * g / (2 * l) * jnp.sin(theta)
            + 3.0 / (m * l**2) * u
        )
        * dt
    )

    newthdot = jnp.clip(newthdot, -max_speed, max_speed)
    newth = theta + newthdot * dt


    theta = newth.reshape(-1,1)
    theta_dot = newthdot.reshape(-1,1)
    x = jnp.cos(theta).reshape(-1,1)
    y = jnp.sin(theta).reshape(-1,1)
   
   

    return jnp.concatenate([x, y,theta_dot], axis=-1)



def get_action_space(env_name,env):
    if env_name == "CartPole-v1":
        return jnp.array([-10.]),jnp.array([10.])

    if env_name == "Pendulum-v1":
        return jnp.array([-2.]),jnp.array([2.])

    else:      
          return jnp.array(env.action_space.low),jnp.array(env.action_space.high)

def get_action_cov(env_name,env,sigma):
    if env_name == "CartPole-v1":
        return jnp.array([5.0])
    if env_name == "Pendulum-v1":
        return jnp.array([2.0])
    else:
        return jnp.array([sigma*1.0])*jnp.ones((env.action_space.shape))

def get_state(state,action=None,time=None,env_name="CartPole-v1"):
    if env_name == "CartPole-v1":
        return CartPoleEnvState(x=state[0],x_dot=state[1],theta=state[2],theta_dot=state[3],time=time)

    if env_name == "Pendulum-v1":
        return PendulumEnvState(x=state[0],y=state[1],theta_dot=state[2],last_u=action,time=time)

    if env_name == "MountainCarContinuous-v0":
        return MountainCar(position=state[0],velocity=state[1],time=time)


def get_step_model(env_name,env):
    if env_name == "CartPole-v1":
        return cartpole_step
    if env_name == "Pendulum-v1":
        return pendulum_step
    if env_name == "MountainCarContinuous-v0":
        return mountaincar_step
    else:
        #return mjx_step
        return mjx_step


@partial(jax.jit, static_argnames=("step_fn",))
def kinematics(state, action, step_fn):
    """Rollout a jitted gymnax episode with lax.scan."""

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""

        action = tmp
        next_state = step_fn(state_input,action)

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





@partial(jax.jit, static_argnames=("gym_env", "frame_skip"))
def kinematics_mujoco(mjx_model, mjx_data, init_state, actions, gym_env, frame_skip=1):
    """
    Functional MJX rollout with lax.scan.

    Args:
        mjx_model: mjx.Model
        mjx_data:  mjx.Data  (initial data, e.g. from reset_mjx_state)
        init_state: (s_dim,) or (batch, s_dim) array of [qpos, qvel]
        actions: (T, a_dim) or (T, batch, a_dim) array
        gym_env: env name (static, unused except for API consistency)
        frame_skip: int, number of mjx.step calls per RL step (static)

    Returns:
        states: (T, s_dim) or (T, batch, s_dim) array of [qpos, qvel]
    """

    # Ensure init_state is used only once: set qpos/qvel at t=0
    def _init_from_state(mjx_data, state):
        return mjx_data.replace(
            qpos=state[..., : mjx_model.nq],
            qvel=state[..., mjx_model.nq : mjx_model.nq + mjx_model.nv],
        )

    mjx_data = _init_from_state(mjx_data, init_state)

    def one_step(carry_data, action_t):
        """
        carry_data: mjx.Data at time t
        action_t: action at time t
        """

        def substep(d, _):
            # Set control, step physics once
            d = d.replace(ctrl=action_t)
            d = mjx.step(mjx_model, d)
            return d, None

        # Perform frame_skip internal physics steps
        carry_data, _ = lax.scan(substep, carry_data, xs=None, length=frame_skip)

        # Observation = [qpos, qvel]
        obs_t = jnp.concatenate([carry_data.qpos, carry_data.qvel], axis=-1)
        return carry_data, obs_t

    # Scan over time dimension
    _, states = lax.scan(one_step, mjx_data, actions)
    return states


def kinematics_simplified_walker(mjx_model, mjx_data, init_state, actions, gym_env, frame_skip=1):
    """
    Simplified Walker2d dynamics rollout with lax.scan interface matching kinematics_mujoco.

    Args:
        mjx_model: Unused (for interface compatibility)
        mjx_data: Unused (for interface compatibility)
        init_state: (s_dim,) or (batch, s_dim) array - initial state
        actions: (T, a_dim) or (T, batch, a_dim) array - action sequence
        gym_env: env name (static, for interface compatibility)
        frame_skip: int, number of physics steps per action (default 4)

    Returns:
        states: (T, s_dim) or (T, batch, s_dim) array of next states
    """
    from src.control.simplified_walker import SimplifiedWalker

    # Create SimplifiedWalker instance
    walker = SimplifiedWalker(dt=0.002, frame_skip=frame_skip)

    # Handle both single and batched actions
    def one_step(state, action):
        """
        state: current state (s_dim,)
        action: action at time t (a_dim,)
        """
        next_state = walker.step(state, action)
        return next_state, next_state

    # Scan over time dimension
    # actions shape: (T, a_dim) or (T, batch, a_dim)
    # init_state shape: (s_dim,) or (batch, s_dim)

    if actions.ndim == 3:
        # Batched: (T, batch, a_dim)
        # Need to vmap over batch dimension
        def batched_step(state_batch, action_batch):
            # state_batch: (batch, s_dim)
            # action_batch: (batch, a_dim)
            next_states = jax.vmap(walker.step)(state_batch, action_batch)
            return next_states, next_states

        _, states = lax.scan(batched_step, init_state, actions)
    else:
        # Single: (T, a_dim)
        _, states = lax.scan(one_step, init_state, actions)

    return states


# This is the FAST batched version (adapted from the first code block)

@partial(jax.jit, static_argnames=("gym_env", "frame_skip"))
def kinematics_mujoco_original(mjx_model, mjx_data, init_state, actions, gym_env, frame_skip=1):
    """
    Functional MJX rollout with lax.scan.

    Args:
        mjx_model: mjx.Model
        mjx_data:  mjx.Data  (initial data, e.g. from reset_mjx_state)
        init_state: (s_dim,) or (batch, s_dim) array of [qpos, qvel]
        actions: (T, a_dim) or (T, batch, a_dim) array
        gym_env: env name (static, unused except for API consistency)
        frame_skip: int, number of mjx.step calls per RL step (static)

    Returns:
        states: (T, s_dim) or (T, batch, s_dim) array of [qpos, qvel]
    """

    # Ensure init_state is used only once: set qpos/qvel at t=0
    def _init_from_state(mjx_data, state):
        return mjx_data.replace(
            qpos=state[..., : mjx_model.nq],
            qvel=state[..., mjx_model.nq : mjx_model.nq + mjx_model.nv],
        )

    mjx_data = _init_from_state(mjx_data, init_state)

    def one_step(carry_data, action_t):
        """
        carry_data: mjx.Data at time t
        action_t: action at time t
        """

        def substep(d, _):
            # Set control, step physics once
            d = d.replace(ctrl=action_t)
            d = mjx.step(mjx_model, d)
            return d, None

        # Perform frame_skip internal physics steps
        carry_data, _ = lax.scan(substep, carry_data, xs=None, length=frame_skip)

        # Observation = [qpos, qvel]
        obs_t = jnp.concatenate([carry_data.qpos, carry_data.qvel], axis=-1)
        return carry_data, obs_t

    # Scan over time dimension
    _, states = lax.scan(one_step, mjx_data, actions)
    return states


def kinematics_simplified_walker(mjx_model, mjx_data, init_state, actions, gym_env, frame_skip=1):
    """
    Simplified Walker2d dynamics rollout with lax.scan interface matching kinematics_mujoco.

    Args:
        mjx_model: Unused (for interface compatibility)
        mjx_data: Unused (for interface compatibility)
        init_state: (s_dim,) or (batch, s_dim) array - initial state
        actions: (T, a_dim) or (T, batch, a_dim) array - action sequence
        gym_env: env name (static, for interface compatibility)
        frame_skip: int, number of physics steps per action (default 4)

    Returns:
        states: (T, s_dim) or (T, batch, s_dim) array of next states
    """
    from src.control.simplified_walker import SimplifiedWalker

    # Create SimplifiedWalker instance
    walker = SimplifiedWalker(dt=0.002, frame_skip=frame_skip)

    # Handle both single and batched actions
    def one_step(state, action):
        """
        state: current state (s_dim,)
        action: action at time t (a_dim,)
        """
        next_state = walker.step(state, action)
        return next_state, next_state

    # Scan over time dimension
    # actions shape: (T, a_dim) or (T, batch, a_dim)
    # init_state shape: (s_dim,) or (batch, s_dim)

    if actions.ndim == 3:
        # Batched: (T, batch, a_dim)
        # Need to vmap over batch dimension
        def batched_step(state_batch, action_batch):
            # state_batch: (batch, s_dim)
            # action_batch: (batch, a_dim)
            next_states = jax.vmap(walker.step)(state_batch, action_batch)
            return next_states, next_states

        _, states = lax.scan(batched_step, init_state, actions)
    else:
        # Single: (T, a_dim)
        _, states = lax.scan(one_step, init_state, actions)

    return states


# This is the FAST batched version (adapted from the first code block)
