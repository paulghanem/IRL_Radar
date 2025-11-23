# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:31:21 2025

@author: siliconsynapse
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax
from mujoco import mjx


@partial(jax.jit, static_argnums=(3,))
def mjx_step_batch(
    mjx_model: mjx.Model,
    data_batch,
    action_batch: jnp.ndarray,
    frame_skip: int,
):
    """
    Batched MJX step for B parallel environments.

    Args:
        mjx_model: MJX-compiled MuJoCo model.
        data_batch: pytree of B mjx.Data objects.
        action_batch: (B, act_dim) action batch.
        frame_skip: number of internal mjx.step calls per RL step.

    Returns:
        Updated data_batch after frame_skip steps.
    """
    def one_env_step(d, u):
        d = d.replace(ctrl=u)
        d = mjx.step(mjx_model, d)
        return d

    def substep(d, _):
        d = jax.vmap(one_env_step, in_axes=(0, 0))(d, action_batch)
        return d, None

    data_batch, _ = lax.scan(substep, data_batch, xs=None, length=frame_skip)
    return data_batch


@jax.jit
def extract_obs_batch(data_batch):
    """
    Extracts observation = [qpos, qvel] from a batch of mjx.Data.

    Args:
        data_batch: pytree of B mjx.Data objects.

    Returns:
        obs_batch: (B, s_dim) where s_dim = nq + nv.
    """
    def obs_one(d):
        return jnp.concatenate([d.qpos, d.qvel], axis=-1)

    return jax.vmap(obs_one, in_axes=0)(data_batch)


def cost_fn(states: jnp.ndarray, state_train) -> jnp.ndarray:
    """
    Compute scalar cost for a batch of states using a Flax TrainState.

    Args:
        states: (N, s_dim)
        state_train: flax.training.train_state.TrainState

    Returns:
        costs: (N,) scalar costs
    """
    out = state_train.apply_fn({'params': state_train.params}, states)
    # Allow (N,1) or (N,) outputs.
    out = jnp.squeeze(out)
    return out.reshape(-1)

import functools
import jax
import jax.numpy as jnp
from jax.random import multivariate_normal

@functools.partial(jax.jit, static_argnames=("num_samples", "horizon"))
def mppi_plan_mujoco(
    mjx_model,
    state,
    prev_action_seq,
    key,
    state_train,
    gail,
    *,
    num_samples,
    horizon,
    u_min,
    u_max,
    sigmas,
    lambda_,
    exploration,
    cost_fn,
):
    """
    Single MPPI planning step for MuJoCo using static num_samples and horizon.
    All *sizes* (num_samples, horizon) are Python ints, passed as static kwargs.
    """

    # --------- ensure state/prev_action_seq are JAX arrays ---------
    state = jnp.asarray(state, dtype=jnp.float32)           # (s_dim,)
    prev_action_seq = jnp.asarray(prev_action_seq, jnp.float32)  # (H, act_dim)

    B = num_samples
    H = horizon
    act_dim = prev_action_seq.shape[-1]

    # --------- build covariance over actions (H, act_dim, act_dim) ---------
    # sigmas: (act_dim,)
    cov_t = jnp.diag(sigmas ** 2)                          # (act_dim, act_dim)
    covariance = jnp.tile(cov_t[None, :, :], (H, 1, 1))    # (H, act_dim, act_dim)
    zero_mean = jnp.zeros((H, act_dim), dtype=jnp.float32)

    # --------- sample noise for all (B, H, act_dim) ---------
    action_noises = multivariate_normal(
        key,
        mean=zero_mean,
        cov=covariance,
        shape=(B,),
    )  # (B, H, act_dim)
    key, _ = jax.random.split(key)

    # --------- build candidate action sequences ---------
    mean_action_seq = prev_action_seq                       # (H, act_dim)

    threshold = int(B * (1.0 - exploration))
    inherited = mean_action_seq + action_noises[:threshold] # (threshold, H, act_dim)
    exploratory = action_noises[threshold:]                 # (B - threshold, H, act_dim)
    perturbed_action_seqs = jnp.concatenate(
        [inherited, exploratory],
        axis=0,
    )  # (B, H, act_dim)

    # clamp to action bounds
    u_min = jnp.asarray(u_min, dtype=jnp.float32)
    u_max = jnp.asarray(u_max, dtype=jnp.float32)
    perturbed_action_seqs = jnp.clip(perturbed_action_seqs, u_min, u_max)

    # --------- roll out dynamics in *state space* only ---------
    # This version assumes you have a *state-only* kinematics for cost_fn.
    # If cost_fn uses full state, we only care about states, not mjx.Data.

    # initial_state_batch: (B, s_dim)
    init_state_batch = jnp.repeat(state[None, :], B, axis=0)

    # a simple placeholder "dynamics" for cost evaluation:
    #   we don't step MuJoCo here, we just feed predicted states into cost_fn.
    #   If you want true physics, you plug in your mjx rollout instead.
    def one_traj_rollout(carry, actions_t):
        # carry: (B, s_dim)
        # actions_t: (B, act_dim)
        # For cost, we often don't need true dynamics; but here's a placeholder:
        next_states = carry  # no-op dynamics; replace with mjx if you wish
        return next_states, next_states

    # transpose actions to (H, B, act_dim) for scan
    actions_TBA = jnp.swapaxes(perturbed_action_seqs, 0, 1)  # (H, B, act_dim)

    _, states_TBS = jax.lax.scan(
        lambda c, a: one_traj_rollout(c, a),
        init_state_batch,
        actions_TBA,
    )  # (H, B, s_dim)

    # states_BHS: (B, H, s_dim)
    states_BHS = jnp.swapaxes(states_TBS, 0, 1)

    # --------- compute MPPI costs ---------
    # flatten states for cost_fn: (B*H, s_dim)
    states_flat = states_BHS.reshape(B * H, -1)

    # cost_fn should be: cost_fn(states_flat, state_train) -> (B*H,)
    costs_flat = cost_fn(states_flat, state_train)
    costs_flat = jnp.asarray(costs_flat).reshape(B, H)

    total_costs = jnp.sum(costs_flat, axis=1)  # (B,)

    # optional GAIL-like transform
    if gail:
        D = jnp.exp(-total_costs) / (jnp.exp(-total_costs) + 1.0)
        total_costs = -jnp.log(D)

    # --------- compute MPPI weights and optimal sequence ---------
    weights = jax.nn.softmax(-total_costs / lambda_, axis=0)  # (B,)

    optimal_action_seq = jnp.sum(
        weights[:, None, None] * perturbed_action_seqs, axis=0
    )  # (H, act_dim)

    return optimal_action_seq, optimal_action_seq, key


class FastMPPI:
    def __init__(self,
                 mjx_model,
                 dim_state,
                 dim_control,
                 u_min,
                 u_max,
                 sigmas,
                 lambda_,
                 horizon,
                 num_samples,
                 frame_skip,
                 exploration,
                 seed,
                 cost_function):       # <-- ADD THIS

        self.mjx_model = mjx_model
        self.dim_state = dim_state
        self.dim_control = dim_control
        self.u_min = u_min
        self.u_max = u_max
        self.sigmas = sigmas
        self.lambda_ = lambda_
        self.horizon = horizon
        self.num_samples = num_samples
        self.frame_skip = frame_skip
        self.exploration = exploration

        self.key = jax.random.PRNGKey(seed)
        self.prev_action_seq = jnp.zeros((horizon, dim_control))

        # Base mjx.Data prototype
        self.base_data = mjx.make_data(mjx_model)
        self.num_samples = num_samples
        self.horizon = horizon
        self.u_min = u_min
        self.u_max = u_max
        self.sigmas = sigmas
        self.lambda_ = lambda_
        self.exploration = exploration
        self.cost_fn = cost_function  # or whatever you passed to MPPI before
        self.prev_action_seq = jnp.zeros((horizon, dim_control), dtype=jnp.float32)


    def reset(self):
        """Reset warm-start sequence."""
        self.prev_action_seq = jnp.zeros((self.horizon, self.dim_control))

    def act(self, state, state_train, gail=False):
        # Make sure state is a JAX array
        state = jnp.asarray(state, dtype=jnp.float32)
    
        action_seq, new_seq, new_key = mppi_plan_mujoco(
            self.mjx_model,
            state,
            self.prev_action_seq,
            self.key,
            state_train,
            gail,
            num_samples=self.num_samples,
            horizon=self.horizon,
            u_min=self.u_min,
            u_max=self.u_max,
            sigmas=self.sigmas,
            lambda_=self.lambda_,
            exploration=self.exploration,
            cost_fn=self.cost_fn,
        )
    
        # Store updated key & action sequence for warm start
        self.key = new_key
        self.prev_action_seq = new_seq
    
        # Return only first action in horizon
        return action_seq[0]



def rollout_mujoco_episode(
    mjx_model,
    fast_mppi: FastMPPI,
    state_train,
    init_state: jnp.ndarray,
    num_steps: int,
    dt: float,
    frame_skip: int,
    reward_fn,         # your reward_fn(gym_env, state, action, next_state, mjx_data, dt, frame_skip)
    gym_env_name: str,
):
    """
    Simple Python-level rollout using FastMPPI + MJX.

    Returns:
        states:  list of states
        actions: list of actions
        rewards: list of rewards
    """
    states = []
    actions = []
    rewards = []

    # Create a fresh data object and inject init_state
    data = mjx.make_data(mjx_model)

    def inject_state_single(d, s):
        return d.replace(
            qpos=s[: mjx_model.nq],
            qvel=s[mjx_model.nq: mjx_model.nq + mjx_model.nv],
        )

    data = inject_state_single(data, init_state)

    state = init_state

    for t in range(num_steps):
        # MPPI control
        action = fast_mppi.act(state, state_train)  # (act_dim,)

        # One environment step via MJX
        data = mjx_step_batch(
            mjx_model,
            jax.tree_util.tree_map(lambda x: x[None, ...], data),  # batch of size 1
            action[None, :],
            frame_skip,
        )
        # Unbatch back to single data
        data = jax.tree_util.tree_map(lambda x: x[0], data)
        next_state = jnp.concatenate([data.qpos, data.qvel], axis=-1)

        r = reward_fn(gym_env_name, state, action, next_state, data, dt, frame_skip)

        states.append(state)
        actions.append(action)
        rewards.append(r)

        state = next_state

    return states, actions, rewards


def generate_session_fast_mppi(
    fast_mppi,
    mjx_model,
    state_train,
    D_demo,
    args,
    reward_fn,
):
    """
    Fast rollout using FastMPPI + MJX.

    This replaces your old generate_session_lax().

    Returns:
        (states, probs, actions, total_reward)
    """
    # ---------------------------------------
    # INITIAL STATE FROM DEMO (like old code)
    # ---------------------------------------
    init_state = D_demo[0, :args.s_dim]

    # Create a single mjx.Data instance
    data = mjx.make_data(mjx_model)

    def inject_state_single(d, s):
        return d.replace(
            qpos=s[: mjx_model.nq],
            qvel=s[mjx_model.nq : mjx_model.nq + mjx_model.nv],
        )

    data = inject_state_single(data, init_state)

    state = init_state
    state = jnp.asarray(state, dtype=jnp.float32)


    # Logging buffers
    states_buf = []
    actions_buf = []
    probs_buf = []          # your code expects prob values
    total_reward = 0.0

    # --------------------------------------
    # ROLLOUT THE EPISODE FOR N STEPS
    # --------------------------------------
    for t in range(args.N_steps):

        # Compute MPPI action
        action = fast_mppi.act(state, state_train)   # (act_dim,)

        # Add to logs
        states_buf.append(state)
        actions_buf.append(action)
        probs_buf.append(1.0)   # MPPI always uses prob=1 in GCL/AIRL

        # Perform MJX step (frame_skip internal)
        data_batch = jax.tree.map(lambda x: x[None, ...], data)      # (1, ...)
        action_batch = action[None, :]

        data_batch = mjx_step_batch(
            fast_mppi.mjx_model,
            data_batch,
            action_batch,
            args.frame_skip
        )

        # unbatch
        data = jax.tree.map(lambda x: x[0], data_batch)

        # new state
        next_state = jnp.concatenate([data.qpos, data.qvel], axis=-1)

        # reward
        r = reward_fn(args.gym_env, state, action, next_state, data, args.dt, args.frame_skip)
        total_reward += float(r)

        state = next_state

    # ---------------------------------------------------------
    # Return like your old code: [states, probs, actions, reward]
    # ---------------------------------------------------------
    states_buf = jnp.array(states_buf)
    actions_buf = jnp.array(actions_buf)
    probs_buf = jnp.array(probs_buf)

    return [states_buf, probs_buf, actions_buf, total_reward]
