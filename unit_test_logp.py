    
import torch
from torch.distributions import Normal, TransformedDistribution, TanhTransform
import os 
import mujoco
from mujoco import mjx 
import gymnasium as gym
import jax.numpy as jnp
import jax
import math  
def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = jnp.sum(-0.5 * jnp.power(noises, 2) - log_stds, axis=-1) - 0.5 * math.log(2 * math.pi) * log_stds.shape[-1]

    return gaussian_log_probs - jnp.sum( jnp.log(1 - jnp.power(actions, 2) + 1e-6), axis=-1, keepdims=True)





for i in range(10):
    mu = -0.1*i*torch.tensor([[0.5, -0.3]])
    log_std = -0.1*i*torch.tensor([[0.1, 0.1]])
    std = log_std.exp()
    base = Normal(mu, std)
    
    key = jax.random.PRNGKey(123)
    key, subkey = jax.random.split(key)
    
    noise = jax.random.normal(subkey, mu.shape)
    u = jnp.array(mu) + jnp.exp(jnp.array(log_std)) * noise
    action=jnp.tanh(u)
    # Built-in torch logp
    
    tanh_normal = TransformedDistribution(base, [TanhTransform(cache_size=1)])
    logp_torch = tanh_normal.log_prob(torch.tensor(action)).sum()
    
    logp=calculate_log_pi(jnp.array(log_std), noise, action)
    
    print(logp_torch,logp)
