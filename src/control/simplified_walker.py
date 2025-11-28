import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

class SimplifiedWalker:
    """
    Linearized approximation of Walker2d.
    Best performing configuration (MSE ~1.92).
    
    State: 18-dim [x, z, theta, joints(6), vx, vz, v_theta, joint_vels(6)]
    Action: 6-dim (joint torques)
    """

    def __init__(self, dt=0.002, frame_skip=4):
        self.dt = dt
        self.frame_skip = frame_skip
        self.total_dt = dt * frame_skip
        
        # Dimensions
        self.state_dim = 18
        self.action_dim = 6

        # --- PHYSICS CONSTANTS ---
        self.gravity = 9.81
        self.mass = 3.53 
        
        # --- TUNED PARAMETERS (Best Performance) ---
        # These values yielded the lowest Actuation MSE (~1.9)
        
        self.linear_damping = 0.1      # Air resistance
        self.angular_damping = 1.0     # Root stability
        
        # Joint Dynamics
        self.joint_damping = 4.0       # Higher damping matches effective inertia
        self.joint_stiffness = 10.0    # Gravity restoring force approximation
        self.torque_gain = 25.0        # Sensitivity to action
        
        self.forward_push_gain = 2.0   # Forward propulsion efficiency
        self.vertical_spring_k = 500.0 # Ground repulsion stiffness
        
        # Limits
        self.action_min = -1.0
        self.action_max = 1.0

    def reset(self, key=None):
        if key is None: key = jax.random.PRNGKey(0)
        state = jnp.zeros(18)
        state = state.at[0].set(0.0)   # x
        state = state.at[1].set(1.25)  # z (height)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        """
        Linear Dynamics: x_next = x + v*dt + F/m * dt^2
        """
        # 1. Setup
        action = jnp.clip(action, self.action_min, self.action_max)
        dt = self.total_dt

        # 2. Unpack State
        # Positions [0-8]
        x, z, theta = state[0], state[1], state[2]
        joint_pos = state[3:9]
        
        # Velocities [9-17]
        vx, vz, v_theta = state[9], state[10], state[11]
        joint_vel = state[12:18]

        # 3. Compute Accelerations (Linear Approximations)

        # A. Joint Acceleration 
        # acc = (Torque * gain) - (Velocity * damping) - (Position * stiffness)
        joint_acc = (action * self.torque_gain) \
                  - (joint_vel * self.joint_damping) \
                  - (joint_pos * self.joint_stiffness)

        # B. Root (Torso) Acceleration
        # Gravity pulls torso down (pendulum effect) + Hip torques react
        hip_torque = action[0] + action[3]
        pendulum_moment = 5.0 * jnp.sin(theta) 
        root_acc = pendulum_moment - hip_torque - (v_theta * self.angular_damping)

        # C. Vertical Acceleration (Gravity + Ground Spring)
        displacement = 1.25 - z
        in_contact = displacement > -0.1 
        
        spring_force = (self.vertical_spring_k * displacement) - (10.0 * vz)
        ground_force = jnp.where(in_contact, spring_force, 0.0)
        
        # Lift from legs extending
        leg_push = (action[1] + action[4]) * 5.0
        
        az = -self.gravity + (ground_force + leg_push) / self.mass

        # D. Forward Acceleration
        # Sum of thigh/leg torques propels x
        forward_force = jnp.sum(action[:4]) * self.forward_push_gain
        ax = (forward_force / self.mass) - (self.linear_damping * vx)

        # 4. Integration (Euler)
        new_vx = vx + ax * dt
        new_vz = vz + az * dt
        new_v_theta = v_theta + root_acc * dt
        new_joint_vel = joint_vel + joint_acc * dt

        new_x = x + new_vx * dt
        new_z = z + new_vz * dt
        new_theta = theta + new_v_theta * dt
        new_joint_pos = joint_pos + new_joint_vel * dt

        # 5. Constraints (Soft floor)
        new_z = jnp.maximum(new_z, 0.0)

        # Pack State
        next_state = jnp.concatenate([
            jnp.array([new_x, new_z, new_theta]),
            new_joint_pos,
            jnp.array([new_vx, new_vz, new_v_theta]),
            new_joint_vel
        ])

        return next_state

    def compute_reward(self, state, action, next_state):
        """Walker2d-v4 Reward Function"""
        forward_reward = (next_state[0] - state[0]) / self.total_dt
        alive_bonus = 1.0
        
        fall_cond = (
            (jnp.abs(next_state[2]) > 1.0) |
            (next_state[1] < 0.8) |
            (next_state[1] > 2.0)
        )
        
        alive_bonus = jnp.where(fall_cond, 0.0, alive_bonus)
        ctrl_cost = 0.001 * jnp.sum(jnp.square(action))

        return forward_reward - ctrl_cost + alive_bonus

    def check_termination(self, state):
        z = state[1]
        angle = state[2]
        fallen = (z < 0.8) | (z > 2.0)
        too_tilted = jnp.abs(angle) > 1.0
        return fallen | too_tilted

# Helper for JIT compatibility in MPPI
@jax.jit
def simplified_walker_step(state, action):
    walker = SimplifiedWalker()
    if state.ndim == 1:
        return walker.step(state, action)
    elif state.ndim == 2:
        if action.ndim == 1:
            return jax.vmap(walker.step, in_axes=(0, None))(state, action)
        else:
            return jax.vmap(walker.step)(state, action)
    return walker.step(state, action)

def get_simplified_walker_dynamics():
    return simplified_walker_step