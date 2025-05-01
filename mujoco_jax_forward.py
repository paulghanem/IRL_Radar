# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 15:23:41 2025

@author: siliconsynapse
"""

import mujoco 
from mujoco import mjx 
from mujoco.mjx._src import dataclasses,forward

import jax.numpy as jnp
import os

assets_dir="/Users/siliconsynapse/anaconda3/envs/rirl/Lib/site-packages/gym/envs/mujoco/assets"
env_xml = "half_cheetah.xml"

model_path=os.path.join(assets_dir,env_xml)

def mjx_step(mjx_model, mjx_data, action):
    """
    JAX-compatible MJX step function.

    Args:
        mjx_model (mjx.Model): The MJX-compiled MuJoCo model.
        mjx_data (mjx.Data): Current simulation data (state).
        action (jnp.ndarray): Control input to apply.

    Returns:
        mjx.Data: Updated simulation state.
    """
    
    # Apply control to mjx_data
    mjx_data = mjx_data.replace(ctrl=action)

    # Advance simulation
    mjx_data = mjx.step(mjx_model, mjx_data)

    return mjx_data

model = mujoco.MjModel.from_xml_path(model_path)
mjx_model = mjx.put_model(model)
#mjx_model=model
mjx_data = mjx.make_data(mjx_model)

action = jnp.zeros(mjx_model.nu)  # match number of actuators


mjx_data = mjx_step(mjx_model, mjx_data, action)
state=jnp.concatenate([mjx_data.qpos[1:], mjx_data.qvel])