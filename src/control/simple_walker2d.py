"""
Simplified Walker2d Environment - Pure JAX Implementation
Same dimensions as real Walker2d but with analytical dynamics (no MJX)
State: 17 dimensions, Action: 6 dimensions
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


@partial(jit, static_argnames=())
def simple_walker2d_step(state, action):
    """
    Simplified Walker2d dynamics - Pure JAX, no MJX.

    State (18-dim):
        [0]: x position (rootx)
        [1]: z height
        [2]: body angle
        [3-8]: joint angles (6)
        [9]: x velocity
        [10]: z velocity
        [11]: angular velocity
        [12-17]: joint velocities (6)

    Action (6-dim): Joint torques

    Returns:
        next_state (18-dim)
    """
    dt = 0.002 * 5  # timestep * frame_skip

    # Store original shape to restore later
    original_shape = state.shape
    # Ensure state and action are 1D arrays for computation
    state_flat = jnp.atleast_1d(state).flatten()
    action_flat = jnp.atleast_1d(action).flatten()

    # Extract state components (18-dim)
    x_pos = state_flat[0]
    z_height = state_flat[1]
    body_angle = state_flat[2]
    joint_angles = state_flat[3:9]

    x_vel = state_flat[9]
    z_vel = state_flat[10]
    ang_vel = state_flat[11]
    joint_vels = state_flat[12:]

    # Simplified dynamics:
    # 1. Actions affect joint accelerations
    # 2. Joint motion affects body motion
    # 3. Gravity and stability constraints

    # Action clipping
    action_flat = jnp.clip(action_flat, -1.0, 1.0)

    # Joint accelerations from actions (simplified)
    joint_accels = action_flat * 10.0  # scale factor

    # Joint velocities update
    new_joint_vels = joint_vels + joint_accels * dt
    new_joint_vels = jnp.clip(new_joint_vels, -10.0, 10.0)  # velocity limits

    # Joint angles update
    new_joint_angles = joint_angles + new_joint_vels * dt
    new_joint_angles = jnp.clip(new_joint_angles, -jnp.pi, jnp.pi)

    # Body motion from joints (simplified coupling)
    # Forward velocity depends on leg motion
    leg_motion = jnp.sum(jnp.abs(new_joint_vels[:3]))  # first 3 joints = legs
    x_accel = leg_motion * 0.5 - x_vel * 0.1  # forward motion with damping

    # Height dynamics (simplified)
    z_accel = -9.81 + jnp.sum(new_joint_vels[1:3]) * 0.3  # gravity + leg push

    # Angular dynamics (body tilt)
    ang_accel = jnp.sum(action_flat[:2]) * 0.5 - ang_vel * 0.5  # from upper body

    # Update velocities
    new_x_vel = x_vel + x_accel * dt
    new_z_vel = z_vel + z_accel * dt
    new_ang_vel = ang_vel + ang_accel * dt

    # Clip velocities
    new_x_vel = jnp.clip(new_x_vel, -10.0, 10.0)
    new_z_vel = jnp.clip(new_z_vel, -10.0, 10.0)
    new_ang_vel = jnp.clip(new_ang_vel, -5.0, 5.0)

    # Update positions
    new_x_pos = x_pos + new_x_vel * dt
    new_z_height = z_height + new_z_vel * dt
    new_body_angle = body_angle + new_ang_vel * dt

    # Keep robot above ground
    new_z_height = jnp.maximum(new_z_height, 0.8)

    # Assemble next state (18-dim)
    next_state = jnp.concatenate([
        jnp.array([new_x_pos, new_z_height, new_body_angle]),
        new_joint_angles,
        jnp.array([new_x_vel, new_z_vel, new_ang_vel]),
        new_joint_vels
    ])

    # Restore original shape if input was not 1D
    if original_shape != (18,):
        next_state = next_state.reshape(original_shape)

    return next_state


@partial(jit, static_argnames=())
def simple_walker2d_reward(state, action, next_state):
    """
    Simplified Walker2d reward function.
    Similar structure to real Walker2d.
    """
    dt = 0.002 * 5

    # Forward reward (velocity in x direction)
    forward_reward = (next_state[0] - state[0]) / dt

    # Alive bonus (stay upright)
    z_height = next_state[1]
    body_angle = next_state[2]

    fall_cond = (
        (jnp.abs(body_angle) > 1.0) |
        (z_height < 0.8) |
        (z_height > 2.0)
    )
    alive_bonus = jnp.where(fall_cond, 0.0, 1.0)

    # Control cost
    ctrl_cost = 0.001 * jnp.sum(jnp.square(action))

    reward = forward_reward - ctrl_cost + alive_bonus

    return reward


@jit
def simple_walker2d_reset():
    """
    Reset to initial state.
    Returns 18-dim state vector.
    """
    # Initial state: standing position
    x_pos = 0.0
    z_height = 1.2
    body_angle = 0.0
    joint_angles = jnp.zeros(6)  # neutral pose

    x_vel = 0.0
    z_vel = 0.0
    ang_vel = 0.0
    joint_vels = jnp.zeros(6)

    state = jnp.concatenate([
        jnp.array([x_pos, z_height, body_angle]),
        joint_angles,
        jnp.array([x_vel, z_vel, ang_vel]),
        joint_vels
    ])

    return state


def get_simple_walker2d_params():
    """
    Get environment parameters for simple Walker2d.
    Matches real Walker2d: 18-dim state (9 qpos + 9 qvel), 6-dim action
    """
    return {
        'state_dim': 18,
        'action_dim': 6,
        'action_min': jnp.array([-1.0] * 6),
        'action_max': jnp.array([1.0] * 6),
        'dt': 0.002,
        'frame_skip': 5,
    }
