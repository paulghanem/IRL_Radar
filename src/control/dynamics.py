from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import jit

from flax import struct
import chex
from typing import Any, Dict, Optional, Tuple, Union
from jax import lax

@struct.dataclass
class CartPoleEnvState(object):
    x: jnp.ndarray
    x_dot: jnp.ndarray
    theta: jnp.ndarray
    theta_dot: jnp.ndarray

def cartpole_step(
        x,x_dot,theta,theta_dot,
        action: Union[int, float, chex.Array],
) -> Tuple[chex.Array, CartPoleEnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
    gravity = 9.8
    masscart = 1.0
    masspole= 0.1
    total_mass = masscart + masspole  # (masscart + masspole)
    length = 0.5
    polemass_length = masspole * length  # (masspole * length)
    force_mag = 10.0
    tau = 0.02

    """Performs step transitions in the environment."""
    force = force_mag * action - force_mag * (1 - action)
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


def cartpole_kinematics(x, x_dot, theta, theta_dot, action, cartpole_step):
    """Rollout a jitted gymnax episode with lax.scan."""

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        x, x_dot, theta, theta_dot = state_input
        a = tmp

        next_x, next_x_dot, next_theta, next_theta_dot = cartpole_step(x, x_dot, theta, theta_dot, a)

        carry = [next_x, next_x_dot, next_theta, next_theta_dot]
        return carry, carry

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        policy_step,
        [x, x_dot, theta, theta_dot],
        action,
    )
    # Return masked sum of rewards accumulated by agent in episode
    x, x_dot, theta, theta_dot = scan_out
    return x, x_dot, theta, theta_dot

