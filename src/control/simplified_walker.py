"""
Simplified Walker2d Dynamics

A linearized approximation of Walker2d with the same state/action dimensions.
State: 17-dim (height, angle, velocities, joint angles, joint velocities)
Action: 6-dim (joint torques)

This is much faster than MuJoCo but maintains the same dimensionality.
"""

import jax
import jax.numpy as jnp
from functools import partial


class SimplifiedWalker:
    """Simplified Walker2d with same dimensions as MuJoCo version"""

    def __init__(self, dt=0.002, frame_skip=4):
        """
        Args:
            dt: Base physics timestep (default 0.002 for Walker2d)
            frame_skip: Number of physics steps per action (default 4 for Walker2d)
        """
        self.dt = dt
        self.frame_skip = frame_skip
        self.state_dim = 18
        self.action_dim = 6

        # State indices (matching Walker2d observation - 18 total)
        # [0]: x (forward position)
        # [1]: z (height - torso height)
        # [2-9]: joint angles (8 angles: rooty, thigh, leg, foot x2)
        # [10-17]: velocities (8 velocities including x_vel as first)

        # Physics parameters
        self.gravity = 9.81

        # Body segment properties - from Walker2d XML
        self.torso_mass = 3.665  # From Walker2d torso
        self.thigh_mass = 4.056  # From Walker2d thigh
        self.leg_mass = 2.781    # From Walker2d leg
        self.foot_mass = 3.161   # From Walker2d foot
        self.total_mass = self.torso_mass + 2 * (self.thigh_mass + self.leg_mass + self.foot_mass)

        # Segment lengths from Walker2d XML (in meters)
        self.torso_length = 0.20   # Torso
        self.thigh_length = 0.225  # Thigh
        self.leg_length = 0.245    # Leg
        self.foot_length = 0.039   # Foot

        # Improved inertia from Walker2d XML
        # Joint inertias (rotational resistance)
        self.joint_inertias = jnp.array([
            4.506,  # Thigh right (from Walker2d XML)
            4.506,  # Leg right
            1.305,  # Foot right
            4.506,  # Thigh left
            4.506,  # Leg left
            1.305   # Foot left
        ])

        # Damping coefficients from Walker2d XML
        self.joint_damping = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Per-joint damping
        self.root_damping = 1.0        # Root angle damping (separate scalar)
        self.ground_damping = 50.0     # Higher for realism
        self.air_damping = 0.1         # Air resistance

        # Ground contact parameters
        self.ground_stiffness = 10000.0  # Stiffer ground
        self.ground_damping_coef = 100.0  # Separate contact damping

        # Friction coefficients
        self.mu_static = 0.9   # Static friction
        self.mu_kinetic = 0.6  # Kinetic friction

        # Actuator parameters (from Walker2d XML)
        # Real motors have nonlinear torque curves
        self.max_motor_torque = 200.0  # Maximum torque
        self.motor_gain = jnp.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])  # Gear ratios

        # Action limits (normalized)
        self.action_min = -1.0
        self.action_max = 1.0

    def reset(self, key=None):
        """Reset to initial state"""
        if key is None:
            key = jax.random.PRNGKey(0)

        # Initial state: standing upright with small noise
        state = jnp.zeros(18)
        state = state.at[0].set(0.0)   # Initial x position
        state = state.at[1].set(1.25)  # Standing height

        # Add small random noise
        noise = jax.random.normal(key, (18,)) * 0.01
        state = state + noise

        return state

    def _compute_foot_positions(self, z, angles):
        """
        Compute foot positions using forward kinematics

        Args:
            z: torso height
            angles: (8,) joint angles

        Returns:
            foot_r_x, foot_r_y, foot_l_x, foot_l_y: positions of both feet
        """
        rooty = angles[0]

        # Right leg: thigh, leg, foot
        thigh_r = angles[2]  # Index 2 after rooty and placeholder
        leg_r = angles[3]
        foot_r = angles[4]

        # Left leg
        thigh_l = angles[5]
        leg_l = angles[6]
        foot_l = angles[7]

        # Right leg forward kinematics (2D)
        # Angles measured from vertical
        hip_r_angle = rooty + thigh_r
        knee_r_angle = hip_r_angle + leg_r
        ankle_r_angle = knee_r_angle + foot_r

        # Thigh endpoint (knee)
        knee_r_x = jnp.sin(hip_r_angle) * self.thigh_length
        knee_r_y = z - jnp.cos(hip_r_angle) * self.thigh_length

        # Leg endpoint (ankle)
        ankle_r_x = knee_r_x + jnp.sin(knee_r_angle) * self.leg_length
        ankle_r_y = knee_r_y - jnp.cos(knee_r_angle) * self.leg_length

        # Foot endpoint
        foot_r_x = ankle_r_x + jnp.sin(ankle_r_angle) * self.foot_length
        foot_r_y = ankle_r_y - jnp.cos(ankle_r_angle) * self.foot_length

        # Left leg forward kinematics
        hip_l_angle = rooty + thigh_l
        knee_l_angle = hip_l_angle + leg_l
        ankle_l_angle = knee_l_angle + foot_l

        knee_l_x = jnp.sin(hip_l_angle) * self.thigh_length
        knee_l_y = z - jnp.cos(hip_l_angle) * self.thigh_length

        ankle_l_x = knee_l_x + jnp.sin(knee_l_angle) * self.leg_length
        ankle_l_y = knee_l_y - jnp.cos(knee_l_angle) * self.leg_length

        foot_l_x = ankle_l_x + jnp.sin(ankle_l_angle) * self.foot_length
        foot_l_y = ankle_l_y - jnp.cos(ankle_l_angle) * self.foot_length

        return foot_r_x, foot_r_y, foot_l_x, foot_l_y

    def _compute_com_position(self, x, z, angles):
        """
        Compute center of mass position from all body segments

        Args:
            x: torso x position
            z: torso z position
            angles: joint angles

        Returns:
            com_x, com_z: center of mass position
        """
        # Torso COM
        torso_com_x = x
        torso_com_z = z - self.torso_com

        # Get foot positions for leg segments
        foot_r_x, foot_r_y, foot_l_x, foot_l_y = self._compute_foot_positions(z, angles)

        # Simplified: weight average of torso and legs
        # Torso has dominant mass
        total_mass = self.total_mass

        com_x = (self.torso_mass * torso_com_x +
                 self.thigh_mass * (foot_r_x * 0.5 + foot_l_x * 0.5)) / total_mass

        com_z = (self.torso_mass * torso_com_z +
                 self.thigh_mass * (foot_r_y * 0.5 + foot_l_y * 0.5)) / total_mass

        return com_x, com_z

    def _single_step_with_dt(self, state, action, dt):
        """
        Single physics step with improved dynamics using specified dt

        Args:
            state: (18,) current state
            action: (6,) joint torques
            dt: timestep to use for integration

        Returns:
            next_state: (18,) next state after one physics step
        """
        # Extract state components
        x = state[0]  # Forward position
        z = state[1]  # Torso height
        angles = state[2:10]  # 8 joint angles
        velocities = state[10:18]  # 8 velocities

        x_vel = velocities[0]
        rooty_vel = velocities[1]
        joint_vels = velocities[2:8]  # 6 actuated joint velocities
        z_vel = velocities[1]  # Reuse for vertical velocity

        # ========== 1. Improved Ground Contact Model ==========
        foot_r_x, foot_r_y, foot_l_x, foot_l_y = self._compute_foot_positions(z, angles)

        # Get foot velocities (approximate using joint velocities)
        # This is a simplification - in real physics we'd use Jacobian
        foot_r_vy = z_vel - 0.3 * joint_vels[1]  # Leg affects foot velocity
        foot_l_vy = z_vel - 0.3 * joint_vels[4]

        # Penetration depth (negative when above ground)
        penetration_r = jnp.maximum(0.0, -foot_r_y)
        penetration_l = jnp.maximum(0.0, -foot_l_y)

        # Improved normal forces with nonlinear stiffness
        # F_normal = k * penetration^1.5 - damping * velocity (Hunt-Crossley model)
        # Exponent 1.5 is more realistic for contact than linear spring
        normal_force_r = jnp.where(
            penetration_r > 1e-6,
            self.ground_stiffness * jnp.power(penetration_r, 1.5) - self.ground_damping_coef * foot_r_vy,
            0.0
        )
        normal_force_l = jnp.where(
            penetration_l > 1e-6,
            self.ground_stiffness * jnp.power(penetration_l, 1.5) - self.ground_damping_coef * foot_l_vy,
            0.0
        )

        # Ensure forces are non-negative
        normal_force_r = jnp.maximum(0.0, normal_force_r)
        normal_force_l = jnp.maximum(0.0, normal_force_l)
        total_normal_force = normal_force_r + normal_force_l

        # Contact flags
        foot_r_contact = penetration_r > 1e-6
        foot_l_contact = penetration_l > 1e-6

        # ========== 2. Nonlinear Actuator Model ==========
        # Real motors have saturation at high torques
        # Use tanh for smooth saturation instead of linear scaling
        # This prevents unrealistic forces at large actions
        desired_torques = action * self.motor_gain

        # Nonlinear saturation curve (smooth tanh)
        # For small actions: nearly linear
        # For large actions: saturates smoothly
        joint_torques = self.max_motor_torque * jnp.tanh(desired_torques / self.max_motor_torque)

        # Damping proportional to velocity (per-joint)
        damping_torques = -self.joint_damping * joint_vels

        # Gravity torques on joints
        # Each joint has gravitational torque proportional to mass and angle
        gravity_torques = jnp.zeros(6)

        # Right leg gravity torques
        gravity_torques = gravity_torques.at[0].set(-self.thigh_mass * self.gravity * 0.05 * jnp.sin(angles[2]))  # Right thigh
        gravity_torques = gravity_torques.at[1].set(-self.leg_mass * self.gravity * 0.05 * jnp.sin(angles[3]))     # Right leg
        gravity_torques = gravity_torques.at[2].set(-self.foot_mass * self.gravity * 0.02 * jnp.sin(angles[4]))    # Right foot

        # Left leg gravity torques
        gravity_torques = gravity_torques.at[3].set(-self.thigh_mass * self.gravity * 0.05 * jnp.sin(angles[5]))  # Left thigh
        gravity_torques = gravity_torques.at[4].set(-self.leg_mass * self.gravity * 0.05 * jnp.sin(angles[6]))     # Left leg
        gravity_torques = gravity_torques.at[5].set(-self.foot_mass * self.gravity * 0.02 * jnp.sin(angles[7]))    # Left foot

        # Total torque
        total_torques = joint_torques + damping_torques + gravity_torques

        # Joint accelerations using real inertias from Walker2d XML
        joint_accels = total_torques / self.joint_inertias

        # Semi-implicit Euler integration (more stable)
        new_joint_vels = joint_vels + joint_accels * dt
        new_joint_vels = jnp.clip(new_joint_vels, -50.0, 50.0)  # Realistic velocity limits

        new_joint_angles = angles[2:8] + new_joint_vels * dt
        # Joint limits from Walker2d XML
        new_joint_angles = jnp.clip(new_joint_angles, -2.8, 2.8)

        # ========== 3. Root Angle Dynamics (rooty) ==========
        # Torso angle affected by leg forces and balance
        # Simplified: tendency to pitch forward/backward based on leg asymmetry
        right_leg_torque = jnp.sum(action[0:3])
        left_leg_torque = jnp.sum(action[3:6])
        leg_asymmetry = right_leg_torque - left_leg_torque

        # Gravity moment (torso wants to rotate if tilted)
        gravity_moment = -0.5 * jnp.sin(angles[0]) * self.gravity

        # Root angular acceleration (use root_damping scalar, not joint_damping array)
        rooty_accel = (leg_asymmetry * 0.1 + gravity_moment - self.root_damping * rooty_vel) / 2.0

        new_rooty_vel = rooty_vel + rooty_accel * dt
        new_rooty_vel = jnp.clip(new_rooty_vel, -5.0, 5.0)

        new_rooty = angles[0] + new_rooty_vel * dt
        new_rooty = jnp.clip(new_rooty, -1.5, 1.5)

        # ========== 4. Improved Forward Motion ==========
        # Horizontal forces from ground reaction forces
        # Force is transmitted through leg geometry when foot is in contact

        # Right leg: compute horizontal component of ground reaction
        # Use Jacobian-like approach: torque contributes to horizontal force based on leg configuration
        # When leg is extended and angled, torques create horizontal push
        right_thigh_angle = angles[2]
        right_leg_angle = angles[3]
        right_total_angle = right_thigh_angle + right_leg_angle

        # Horizontal force contribution from right leg torques (when in contact)
        # sin(angle) gives horizontal component, torque/length gives force magnitude
        right_leg_horizontal = jnp.where(
            foot_r_contact,
            (joint_torques[0] * jnp.sin(right_thigh_angle) / self.thigh_length +
             joint_torques[1] * jnp.sin(right_total_angle) / self.leg_length +
             joint_torques[2] * jnp.sin(right_total_angle) / self.foot_length) / 3.0,
            0.0
        )

        # Left leg: same calculation
        left_thigh_angle = angles[5]
        left_leg_angle = angles[6]
        left_total_angle = left_thigh_angle + left_leg_angle

        left_leg_horizontal = jnp.where(
            foot_l_contact,
            (joint_torques[3] * jnp.sin(left_thigh_angle) / self.thigh_length +
             joint_torques[4] * jnp.sin(left_total_angle) / self.leg_length +
             joint_torques[5] * jnp.sin(left_total_angle) / self.foot_length) / 3.0,
            0.0
        )

        # Total horizontal force from legs
        leg_horizontal_force = right_leg_horizontal + left_leg_horizontal

        # Friction constraint: horizontal force limited by friction cone
        # Maximum horizontal force = mu * normal_force
        max_friction = self.mu_kinetic * total_normal_force

        # Air drag (quadratic with velocity)
        air_drag = -0.1 * x_vel * jnp.abs(x_vel)

        # Total horizontal force (constrained by friction)
        total_horizontal_force = jnp.clip(leg_horizontal_force, -max_friction, max_friction) + air_drag

        # Horizontal acceleration
        x_accel = total_horizontal_force / self.total_mass

        # Semi-implicit Euler
        new_x_vel = x_vel + x_accel * dt
        new_x_vel = jnp.clip(new_x_vel, -20.0, 20.0)
        new_x = x + new_x_vel * dt

        # ========== 5. Improved Vertical Motion ==========
        # Vertical ground reaction force from spring-damper model
        # Already computed as total_normal_force

        # Vertical acceleration
        z_accel = (total_normal_force / self.total_mass) - self.gravity

        # Air resistance on vertical motion
        z_accel += -self.air_damping * z_vel

        # Semi-implicit Euler
        new_z_vel = z_vel + z_accel * dt
        new_z_vel = jnp.clip(new_z_vel, -15.0, 15.0)

        new_z = z + new_z_vel * dt
        # Ensure doesn't go below ground (with slight penetration allowed)
        new_z = jnp.maximum(new_z, 0.5)

        # ========== 6. Construct New State ==========
        # State: [x, z, 8 angles, 8 velocities] = 18 dims total
        # Explicitly construct with correct shapes
        new_angles = jnp.concatenate([
            jnp.array([new_rooty]),  # 1: root angle
            jnp.array([0.0]),        # 1: placeholder
            new_joint_angles         # 6: joint angles
        ])  # Total: 1 + 1 + 6 = 8 angles

        # Velocities: [x_vel, z_vel, 6 joint vels] = 8 total
        new_velocities = jnp.concatenate([
            jnp.array([new_x_vel]),  # 1: horizontal velocity
            jnp.array([new_z_vel]),  # 1: vertical velocity
            new_joint_vels           # 6: joint velocities
        ])  # Total: 1 + 1 + 6 = 8 velocities

        new_state = jnp.concatenate([
            jnp.array([new_x]),      # 1 dim: x position
            jnp.array([new_z]),      # 1 dim: z position
            new_angles,              # 8 dims: angles
            new_velocities           # 8 dims: velocities
        ])  # Total: 1 + 1 + 8 + 8 = 18 dims

        return new_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        """
        Simplified dynamics with frame_skip
        Uses effective timestep = dt * frame_skip for efficiency

        Args:
            state: (18,) current state
            action: (6,) joint torques

        Returns:
            next_state: (18,) next state after frame_skip steps
        """
        # Clip actions to valid range
        action = jnp.clip(action, self.action_min, self.action_max)

        # Use effective timestep instead of looping
        # Temporarily increase dt for this step
        original_dt = self.dt
        effective_dt = self.dt * self.frame_skip

        # Create a modified version with larger dt (avoiding mutation)
        # We'll inline the single step with effective_dt
        return self._single_step_with_dt(state, action, effective_dt)

    def compute_reward(self, state, action, next_state):
        """
        Compute reward matching Walker2d (from unit_test.py):
        - Forward velocity (main objective)
        - Alive bonus (stay upright)
        - Control cost (penalize large actions)

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state

        Returns:
            reward: Scalar reward
        """
        # Forward velocity reward - matching unit_test.py calculation
        # forward_reward = (next_state[0] - state[0]) / (dt * frame_skip)
        forward_reward = (next_state[0] - state[0]) / (self.dt * self.frame_skip)

        # Alive bonus - exactly matching unit_test.py
        alive_bonus = 1.0

        # JAX boolean: True when robot falls
        fall_cond = (
            (jnp.abs(next_state[2]) > 1.0) |
            (next_state[1] < 0.8) |
            (next_state[1] > 2.0)
        )

        # If fall_cond is True → alive_bonus = 0
        alive_bonus = jnp.where(fall_cond, 0.0, alive_bonus)

        # Control cost
        ctrl_cost = 0.001 * jnp.sum(jnp.square(action))

        # Total reward
        r = forward_reward - ctrl_cost + alive_bonus

        return r

    def check_termination(self, state):
        """
        Check if episode should terminate

        Args:
            state: Current state

        Returns:
            done: Boolean indicating termination
        """
        z = state[1]
        angle = state[3]

        # Terminate if fallen or too tilted
        fallen = (z < 0.8) | (z > 2.0)
        too_tilted = jnp.abs(angle) > 1.0

        done = fallen | too_tilted

        return done


@jax.jit
def simplified_walker_step(state, action):
    """
    JIT-compiled simplified Walker2d step function
    Compatible with existing MPPI interface

    Args:
        state: (18,) or (N, 18) state
        action: (6,) or (N, 6) action

    Returns:
        next_state: Same shape as state
    """
    walker = SimplifiedWalker()

    # Handle both single and batched inputs
    if state.ndim == 1 and action.ndim == 1:
        # Both single: (18,) and (6,)
        return walker.step(state, action)
    elif state.ndim == 2 and action.ndim == 1:
        # State batched, action not: (N, 18) and (6,)
        # Broadcast action to all states
        return jax.vmap(walker.step, in_axes=(0, None))(state, action)
    elif state.ndim == 2 and action.ndim == 2:
        # Both batched: (N, 18) and (N, 6)
        return jax.vmap(walker.step)(state, action)
    else:
        # Fallback: squeeze and retry
        state = jnp.squeeze(state) if state.ndim > 1 and state.shape[0] == 1 else state
        action = jnp.squeeze(action) if action.ndim > 1 and action.shape[0] == 1 else action
        return walker.step(state, action)


# For compatibility with dynamics.py interface
def get_simplified_walker_dynamics():
    """Returns the step function for simplified Walker2d"""
    return simplified_walker_step
