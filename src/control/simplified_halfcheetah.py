"""
Simplified HalfCheetah Dynamics

A linearized approximation of HalfCheetah with the same state/action dimensions.
State: 17-dim (rootx, rooty, angle, joint angles, velocities)
Action: 6-dim (joint torques)

This is much faster than MuJoCo but maintains the same dimensionality.
"""

import jax
import jax.numpy as jnp
from functools import partial


class SimplifiedHalfCheetah:
    """Simplified HalfCheetah with same dimensions as MuJoCo version"""

    def __init__(self, dt=0.05):
        """
        Args:
            dt: Time step (default matches HalfCheetah frame_skip * dt)
        """
        self.dt = dt
        self.state_dim = 18
        self.action_dim = 6

        # State indices (matching HalfCheetah observation - 18 total)
        # [0-8]: position coordinates (rootx, rooty, angle, 6 joint angles) = 9 dims
        # [9-17]: velocities (9 velocities)

        # Gravity and physics parameters
        self.gravity = 9.81
        self.mass = 1.0

        # Action limits (torque limits)
        self.action_min = -1.0
        self.action_max = 1.0

    def reset(self, key=None):
        """Reset to initial state"""
        if key is None:
            key = jax.random.PRNGKey(0)

        # Initial state: neutral position
        state = jnp.zeros(18)
        state = state.at[0].set(0.0)  # Root X position
        state = state.at[1].set(0.0)  # Root Y at ground level
        state = state.at[2].set(0.0)  # Root angle neutral

        # Add small random noise
        noise = jax.random.normal(key, (18,)) * 0.01
        state = state + noise

        return state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        """
        Simplified dynamics: Linear approximation

        Args:
            state: (18,) current state
            action: (6,) joint torques

        Returns:
            next_state: (18,) next state
        """
        # Clip actions to valid range
        action = jnp.clip(action, self.action_min, self.action_max)

        # Extract state components
        # Positions: rootx(0), rooty(1), angle(2), joint1-6(3-8) = 9 total
        # Velocities: 9 velocities (9-17)
        positions = state[:9]
        velocities = state[9:18]

        rootx, rooty, angle = positions[0], positions[1], positions[2]
        joint_angles = positions[3:9]  # 6 joint angles

        rootx_vel, rooty_vel, angle_vel = velocities[0], velocities[1], velocities[2]
        joint_vels = velocities[3:9]  # 6 joint velocities

        # Simplified dynamics:
        # 1. Joint accelerations from torques
        joint_accels = action * 10.0  # Direct torque influence

        # 2. Update joint velocities
        new_joint_vels = joint_vels + joint_accels * self.dt
        new_joint_vels = jnp.clip(new_joint_vels, -10.0, 10.0)  # Velocity limits

        # 3. Update joint angles
        new_joint_angles = joint_angles + new_joint_vels * self.dt
        new_joint_angles = jnp.clip(new_joint_angles, -jnp.pi, jnp.pi)

        # 4. Root X velocity (forward motion) influenced by leg coordination
        # HalfCheetah runs forward by coordinated leg thrusts
        leg_thrust = jnp.sum(action * jnp.array([0.3, 0.5, 0.3, 0.5, 0.2, 0.2]))
        x_accel = leg_thrust * 1.0 - 0.05 * rootx_vel  # Thrust minus friction
        new_rootx_vel = rootx_vel + x_accel * self.dt
        new_rootx_vel = jnp.clip(new_rootx_vel, -10.0, 10.0)

        # 5. Root Y (height) - simplified ground contact
        # Height oscillates slightly with running motion
        height_change = 0.01 * jnp.sin(jnp.sum(new_joint_angles))
        new_rooty_vel = height_change / self.dt
        new_rooty_vel = jnp.clip(new_rooty_vel, -2.0, 2.0)

        # 6. Root angle (pitch) - influenced by leg imbalance
        front_legs = action[:3]
        back_legs = action[3:]
        imbalance = jnp.sum(front_legs) - jnp.sum(back_legs)
        angle_accel = imbalance * 0.5 - 0.2 * angle_vel  # Damping
        new_angle_vel = angle_vel + angle_accel * self.dt
        new_angle_vel = jnp.clip(new_angle_vel, -5.0, 5.0)

        # Update positions
        new_rootx = rootx + new_rootx_vel * self.dt
        new_rooty = rooty + new_rooty_vel * self.dt
        new_rooty = jnp.maximum(new_rooty, -0.5)  # Don't go too far below ground
        new_angle = angle + new_angle_vel * self.dt
        new_angle = jnp.clip(new_angle, -jnp.pi/2, jnp.pi/2)

        # Construct new state
        new_positions = jnp.concatenate([
            jnp.array([new_rootx, new_rooty, new_angle]),  # 3 root coords
            new_joint_angles                                 # 6 joint angles
        ])  # Total: 9 dims

        new_velocities = jnp.concatenate([
            jnp.array([new_rootx_vel, new_rooty_vel, new_angle_vel]),  # 3 root vels
            new_joint_vels                                               # 6 joint vels
        ])  # Total: 9 dims

        new_state = jnp.concatenate([
            new_positions,   # 9 dims
            new_velocities   # 9 dims
        ])  # Total: 18 dims

        return new_state

    def compute_reward(self, state, action, next_state):
        """
        Compute reward matching HalfCheetah:
        - Forward velocity (main objective)
        - Control cost (penalize large actions)

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            reward: Scalar reward
        """
        # Forward velocity reward (HalfCheetah's main objective)
        forward_vel = next_state[8]  # rootx_vel at index 8
        forward_reward = forward_vel

        # Control cost
        ctrl_cost = 0.1 * jnp.sum(jnp.square(action))

        # Total reward
        reward = forward_reward - ctrl_cost

        return reward

    def check_termination(self, state):
        """
        Check if episode should terminate
        HalfCheetah typically doesn't have early termination

        Args:
            state: Current state

        Returns:
            done: Boolean indicating termination
        """
        # HalfCheetah usually doesn't terminate early
        # But we can terminate if completely fallen over
        rooty = state[1]
        angle = state[2]

        # Terminate only if completely collapsed
        fallen = (rooty < -1.0) | (jnp.abs(angle) > jnp.pi/2)

        return fallen


@jax.jit
def simplified_halfcheetah_step(state, action):
    """
    JIT-compiled simplified HalfCheetah step function
    Compatible with existing MPPI interface

    Args:
        state: (18,) or (N, 18) state
        action: (6,) or (N, 6) action

    Returns:
        next_state: Same shape as state
    """
    cheetah = SimplifiedHalfCheetah()

    # Handle both single and batched inputs
    if state.ndim == 1 and action.ndim == 1:
        # Both single: (18,) and (6,)
        return cheetah.step(state, action)
    elif state.ndim == 2 and action.ndim == 1:
        # State batched, action not: (N, 18) and (6,)
        # Broadcast action to all states
        return jax.vmap(cheetah.step, in_axes=(0, None))(state, action)
    elif state.ndim == 2 and action.ndim == 2:
        # Both batched: (N, 18) and (N, 6)
        return jax.vmap(cheetah.step)(state, action)
    else:
        # Fallback: squeeze and retry
        state = jnp.squeeze(state) if state.ndim > 1 and state.shape[0] == 1 else state
        action = jnp.squeeze(action) if action.ndim > 1 and action.shape[0] == 1 else action
        return cheetah.step(state, action)


# For compatibility with dynamics.py interface
def get_simplified_halfcheetah_dynamics():
    """Returns the step function for simplified HalfCheetah"""
    return simplified_halfcheetah_step
