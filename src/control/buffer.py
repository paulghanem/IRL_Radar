# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 12:29:10 2025

Converted to JAX + Flax by ChatGPT
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import flax
from flax import struct
from typing import Tuple
import numpy as np
import os
import pdb


# ======================================================
# 📦 Base serialized buffer (for loading from disk)
# ======================================================

@struct.dataclass
class SerializedBuffer:
    states: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    next_states: jnp.ndarray

    @property
    def buffer_size(self):
        return self.states.shape[0]

    @classmethod
    def load(cls, path: str, device=jax.devices("cpu")[0]) -> SerializedBuffer:
        """Load from torch file or npz (depending on how saved)."""
        import torch
        tmp = torch.load(path, map_location="cpu")
        return cls(
            states=jnp.array(tmp["state"]),
            actions=jnp.array(tmp["action"]),
            rewards=jnp.array(tmp["reward"]),
            dones=jnp.array(tmp["done"]),
            next_states=jnp.array(tmp["next_state"]),
        )

    def sample(self, key: jax.random.PRNGKey, batch_size: int):
        idx = jax.random.randint(key, (batch_size,), 0, self.buffer_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.next_states[idx],
        )

    def get_sample(self, idx: jnp.ndarray):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.next_states[idx],
        )


# ======================================================
# 🧠 Circular Replay Buffer
# ======================================================

@struct.dataclass
class Buffer:
    states: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    next_states: jnp.ndarray
    n: int
    p: int
    buffer_size: int

    @classmethod
    def create(cls, buffer_size: int, state_shape: Tuple[int], action_shape: Tuple[int]):
        zeros_s = jnp.zeros((buffer_size, *state_shape))
        zeros_a = jnp.zeros((buffer_size, *action_shape))
        zeros_r = jnp.zeros((buffer_size, 1))
        zeros_d = jnp.zeros((buffer_size, 1))
        zeros_ns = jnp.zeros((buffer_size, *state_shape))
        return cls(zeros_s, zeros_a, zeros_r, zeros_d, zeros_ns, n=0, p=0, buffer_size=buffer_size)

    def append(
        self, state, action, reward, done, next_state
    ) -> Buffer:
        """Return new buffer with appended transition."""
        p = self.p
        states = self.states.at[p].set(state)
        actions = self.actions.at[p].set(action)
        rewards = self.rewards.at[p].set(reward)
        dones = self.dones.at[p].set(done)
        next_states = self.next_states.at[p].set(next_state)

        p_new = (p + 1) % self.buffer_size
        n_new = jnp.minimum(self.n + 1, self.buffer_size)
        return self.replace(states=states, actions=actions, rewards=rewards,
                            dones=dones, next_states=next_states, p=p_new, n=n_new)

    def sample(self, key: jax.random.PRNGKey, batch_size: int):
        idx = jax.random.randint(key, (batch_size,), 0, self.n)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.next_states[idx],
        )


# ======================================================
# 🔁 RolloutBuffer (used for PPO)
# ======================================================

@struct.dataclass
class RolloutBuffer:
    states: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    log_pis: jnp.ndarray
    next_states: jnp.ndarray
    n: int
    p: int
    buffer_size: int
    total_size: int

    @classmethod
    def create(
        cls, buffer_size: int, state_shape: Tuple[int], action_shape: Tuple[int], mix: int = 1
    ):
        total_size = mix * buffer_size
        zeros_s = jnp.zeros((total_size, *state_shape))
        zeros_a = jnp.zeros((total_size, *action_shape))
        zeros_r = jnp.zeros((total_size, 1))
        zeros_d = jnp.zeros((total_size, 1))
        zeros_lp = jnp.zeros((total_size, 1))
        zeros_ns = jnp.zeros((total_size, *state_shape))
        return cls(zeros_s, zeros_a, zeros_r, zeros_d, zeros_lp, zeros_ns,
                   n=0, p=0, buffer_size=buffer_size, total_size=total_size)

    def append(self, state, action, reward, done, log_pi, next_state) -> RolloutBuffer:
        """Append one transition immutably."""
        p = self.p
        states = self.states.at[p].set(state)
        actions = self.actions.at[p].set(action)
        rewards = self.rewards.at[p].set(reward)
        dones = self.dones.at[p].set(done)
        log_pis = self.log_pis.at[p].set(log_pi)
        next_states = self.next_states.at[p].set(next_state)
        
        p_new = (p + 1) % self.total_size
        n_new = jnp.minimum(self.n + 1, self.total_size)
        return self.replace(states=states, actions=actions, rewards=rewards, dones=dones,
                            log_pis=log_pis, next_states=next_states, p=p_new, n=n_new)

    def get(self):
        """Return last full rollout segment."""
       # pdb.set_trace()
        assert self.p % self.buffer_size == 0, "Buffer not aligned to rollout size"
        start = (self.p - self.buffer_size) % self.total_size
        idx = jnp.arange(start, start + self.buffer_size) % self.total_size
        #pdb.set_trace()
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.log_pis[idx],
            self.next_states[idx],
        )

    def sample(self, key: jax.random.PRNGKey, batch_size: int):
        idx = jax.random.randint(key, (batch_size,), 0, self.n)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.log_pis[idx],
            self.next_states[idx],
        )

    def get_sample(self, idx: jnp.ndarray):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.log_pis[idx],
            self.next_states[idx],
        )
