from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import jit

v_low,v_high = -50.0,50.0
av_low,av_high = -2*jnp.pi,2*jnp.pi

va_low,va_high = -35.0,35.0
ava_low,ava_high = -1*jnp.pi,1*jnp.pi

UNI_SI_U_LIM = jnp.array([[v_low,av_low],[v_high,av_high]])
UNI_DI_U_LIM = jnp.array([[va_low,ava_low],[va_high,ava_high]])


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
def state_multiple_update(p,U,chi,time_step_sizes):
    # sensor dynamics for unicycle model

    vs,avs = U[:,[0]],U[:,[1]]

    chi = chi.reshape(1,1)
    p = p.reshape(1,-1)

    chi_next = chi + jnp.cumsum(time_step_sizes * avs,axis=0)
    chis = jnp.vstack((chi,chi_next))

    ps_next = p + jnp.cumsum(jnp.column_stack((jnp.cos(chis[:-1].ravel()),
                                               jnp.sin(chis[:-1].ravel()))) * vs * time_step_sizes,axis=0)
    ps = jnp.vstack((p,ps_next))

    # chis = [jnp.expand_dims(chi,0)] + [None]*len(vs)
    # ps = [jnp.expand_dims(p,0)] + [None]*len(vs)
    #
    # for k in range(len(vs)):
    #     chi_next = chi + time_step_sizes * avs[k]
    #     p_next = p + time_step_sizes * jnp.array([[jnp.cos(chi.squeeze()),jnp.sin(chi.squeeze())]]) * vs[k]
    #
    #     ps[k+1] = jnp.expand_dims(p_next,0)
    #     chis[k+1] = jnp.expand_dims(chi_next,0)
    #
    #     chi = chi_next
    #     p = p_next
    # chis = jnp.hstack((chi,chi+jnp.cumsum(time_)))
    # p = p.reshape(-1,2)
    # chi = chi.reshape(1,1)

    return ps[-1,:],chis[-1,:],ps,chis



def unicycle_kinematics_single_integrator(U,unicycle_state,dt):
    # sensor dynamics for unicycle model
    horizon = U.shape[-2]
    p,chi = unicycle_state[...,:3],unicycle_state[...,[3]]

    chi = jnp.expand_dims(chi,axis=-1)
    p =  jnp.expand_dims(p,axis=-2)

    vs,avs = jnp.clip(U[...,0],v_low,v_high),jnp.clip(U[...,[1]],av_low,av_high)

    vs = jnp.expand_dims(vs,axis=-1)
    chi_next = chi + jnp.cumsum(dt * avs,axis=-2)
    chis = jnp.concatenate((chi,chi_next),axis=-2)

    chis_temp = chis[...,:-1,[0]]
    ps_next = p + jnp.cumsum(jnp.concatenate((jnp.cos(chis_temp),
                                               jnp.sin(chis_temp),
                                               jnp.zeros(chis_temp.shape)),axis=-1) * vs * dt,axis=-2)
    ps = jnp.concatenate((p,ps_next),axis=-2)

    return jnp.concatenate((ps,chis),axis=-1)

def next_state_fn(current_state, control_input):

    current_pos_x = current_state[...,[0]]
    current_pos_y = current_state[...,[1]]
    current_pos_z = current_state[...,[2]]

    current_ang = current_state[...,[3]]
    current_vel = current_state[...,[4]]
    current_ang_vel = current_state[...,[5]]
    dt = current_state[...,[6]]

    input_acc = jnp.clip(control_input[...,[0]],va_low,va_high)
    input_ang_acc = jnp.clip(control_input[...,[1]],ava_low,ava_high)

    next_vel = current_vel + dt*input_acc
    next_ang_vel = current_ang_vel + dt*input_ang_acc

    next_pos_x = current_pos_x + jnp.cos(current_ang)*current_vel*dt + jnp.cos(current_ang)*.5*input_acc*dt**2
    next_pos_y = current_pos_y + jnp.sin(current_ang)*current_vel*dt + jnp.sin(current_ang)*.5*input_acc*dt**2
    next_pos_z = current_pos_z

    next_ang = current_ang + current_ang_vel*dt + .5 * input_ang_acc*dt**2

    next_vel = jnp.clip(next_vel, v_low, v_high)
    next_ang_vel = jnp.clip(next_ang_vel, av_low, av_high)

    next_state = jnp.concatenate((next_pos_x, next_pos_y, next_pos_z, next_ang, next_vel, next_ang_vel, dt), axis=-1)

    return next_state, next_state

@jit
def unicycle_kinematics_double_integrator(U, unicycle_state, dt):
    horizon = U.shape[-2]

    U = jnp.moveaxis(U,source=-2,destination=0)

    dt_shape = unicycle_state.shape[:-1] + (1,)
    dt_tile = jnp.tile(dt,dt_shape)
    current_state = jnp.concatenate((unicycle_state,dt_tile),axis=-1)

    last_state,next_states = jax.lax.scan(next_state_fn, current_state, U)

    next_states = jnp.moveaxis(next_states[...,:-1],source=0,destination=-2)

    all_states = jnp.concatenate((jnp.expand_dims(unicycle_state,-2),next_states), axis=-2)

    return all_states

