# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:31:52 2024

@author: siliconsynapse
"""

from jax import config

config.update("jax_enable_x64", True)

import jax

import jax.numpy as jnp
from jax import jit


# def rotational_matrix(chi):
#     return jnp.array([[jnp.cos(chi), -jnp.sin(chi)],
#                       [jnp.sin(chi), jnp.cos(chi) ]])
# def rotational_column_perp(chi):
#     return jnp.vstack((-jnp.sin(chi),jnp.cos(chi)))
#
# def rotational_column(chi):
#     return jnp.vstack((jnp.cos(chi),jnp.sin(chi)))
#
# # @jit
# def angle_update(chi,av,time_step_size):
#     return chi + time_step_size*av
#
# def position_update(p,v,av,chi,chi_,time_step_size):
#
#     p = p.T
#
#     return jnp.where(av==0,
#                      p + v *rotational_column(chi)*time_step_size,
#                      p + (v / jnp.where(av == 0., 1e-10, av)) * (rotational_column_perp(chi) - rotational_column_perp(chi_))
#                      ).T
#
# @jit
# def state_multiple_update(p,U,chi,time_step_sizes):
#     vs,avs = U[0,:],U[1,:]
#     chis = [jnp.expand_dims(chi,0)] + [None]*len(vs)
#     ps = [jnp.expand_dims(p,0)] + [None]*len(vs)
#
#
#
#     for k in range(len(vs)):
#         # update angle
#         chi_next = angle_update(chi,avs[k],time_step_sizes[k])
#         chis[k+1] = jnp.expand_dims(chi_next, 0)
#
#         # update position on angle
#         p_next = position_update(p,vs[k],avs[k],chi,chi_next,time_step_sizes[k])
#         ps[k+1] = jnp.expand_dims(p_next,0)
#
#         # reinit for next state
#         chi = chi_next
#         p = p_next
#
#     return p,chi,jnp.vstack(ps),jnp.vstack(chis)

# @jit
def state_target_multiple_update(p,v,U,time_step_sizes):
    # sensor dynamics for second order integrator model

    horizon=U.shape[0]
    ps=p
    vs=v
    
    for i in range(horizon):
        a=U[i,:]
        p_next = p + time_step_sizes * v
        ps = jnp.vstack((ps,p_next))

        v_next =v + time_step_sizes * a
        jnp.clip(v_next,0,50)
        vs = jnp.vstack((vs,v_next))
        
        p=p_next
        v=v_next


    return ps[-1,:],vs[-1,:],ps,vs
