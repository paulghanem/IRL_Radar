import numpy as np
from scipy.spatial import distance_matrix
from jax import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import jit
from jax.tree_util import Partial as partial
from jax import vmap

import matplotlib.pyplot as plt
from sklearn.covariance import ledoit_wolf

# from src.Measurement import RadarEqnMeasure,ExponentialDecayMeasure
# from jax import jacfwd
# l = jacfwd(RadarEqnMeasure)(qs,ps,Pt,Gt,Gr,L,lam,rcs)
def MPPI_wrapper(kinematic_model):

    MPPI_paths_vmap = vmap(kinematic_model, (0, None))
    @jit
    def MPPI(U_nominal,U_MPPI,state):
        # U_nominal is TxC
        # U_MPPI is N_rolloutxTxC

        N_rollout,T,da = U_MPPI.shape

        states = kinematic_model(U_nominal,state)
        states_MPPI = MPPI_paths_vmap(U_MPPI, state)

        return states,states_MPPI

    return MPPI



def MPPI_ptb(stds,N, time_steps, num_traj, key,method="CartPole-Normal"):

    if method=='CartPole-Normal':
        f_min, f_max = stds[0]

        U_force = jax.random.normal(key,shape=(num_traj, N, time_steps,1)) * f_max
        U_ptb = U_force

    return U_ptb

def weighting(method="CE"):

    if method == "CE":
        def weight_fn(costs,elite_threshold=0.8):
            num_traj = costs.shape[0]

            zeta = jnp.round(num_traj * (1-elite_threshold))
            score_zeta = jnp.quantile(costs,1-elite_threshold)

            weight = 1/zeta * (costs <= score_zeta)
            return weight/jnp.sum(weight,0)


    elif method == "information":
        def weight_fn(costs,temperature):

            weight = jax.nn.softmax(-1/temperature * (costs-jnp.min(costs,axis=0)),axis=0)
            return weight

    return weight_fn


def MPPI_scores_wrapper(score_fn,method="gymenv"):


    if method == "gymenv":
        @jit
        def MPPI_scores(U_MPPI,state):
            # the lower the value, the better
            score_fn_partial = partial(score_fn, state=state)

            # vmap over the ACTION INPUT FIRST DIM. THUS ITS MEANT FOR MPPI N x T x da
            MPPI_score_fn = vmap(score_fn_partial)

            costs = MPPI_score_fn(U_MPPI)

            return costs
        
    if method == "NN":
        @jit
        def MPPI_scores(radar_states,target_states,U_MPPI,chis,time_step_size,A,J,gamma,state_train):
            # the lower the value, the better
            score_fn_partial = partial(score_fn,chis=chis, radar_states=radar_states, target_states=target_states, time_step_size=time_step_size,
                                                    A=A,J=J,
                                                    gamma=gamma,state_train=state_train)
            MPPI_score_fn = vmap(score_fn_partial)
            scores = MPPI_score_fn(U_MPPI)

            return scores
        
    if method == "NN_t":
        @jit
        def MPPI_scores(radar_states,target_states,U_MPPI,chis,time_step_size,A,J,gamma,state_train):
            # the lower the value, the better
            score_fn_partial = partial(score_fn,chis=chis, radar_states=radar_states, target_states=target_states, time_step_size=time_step_size,
                                                    A=A,J=J,
                                                    gamma=gamma,state_train=state_train)
            MPPI_score_fn = vmap(score_fn_partial)
            scores = MPPI_score_fn(U_MPPI)

            return scores
        
  


    return MPPI_scores

from copy import deepcopy

def MPPI_control(
                 state,U,cov,key, # radar state
                 kinematic_model, control_constraints,# kinematic model and cubature kalman filter
                 MPPI_kinematics,MPPI_scores,weight_fn, # MPPI need parameters
                 args):

    # horizon x 2
    U_prime = deepcopy(U)
    cov_prime = deepcopy(cov)

    horizon,a_dim = U_prime.shape

    for mppi_iter in range(args.MPPI_iterations):
        key, subkey = jax.random.split(key)


        try:
            E = jax.random.multivariate_normal(key, mean=jnp.zeros_like(U).ravel(), cov=cov_prime,
                                               shape=(args.num_traj,))  # ,method="svd")
        except:
            E = jax.random.multivariate_normal(key, mean=jnp.zeros_like(U).ravel(), cov=cov_prime,
                                               shape=(args.num_traj,), method="svd")

        # simulate the model with the trajectory noise samples
        # number of traj x number of radars x horizon x 2
        V = U_prime + E.reshape(args.num_traj, horizon,a_dim)


        # mppi_sample_end = time()

        # number of radars x horizon+1 x dn
        # number of traj x number of radars x horizon+1 x dn
        states, states_MPPI = MPPI_kinematics(U_nominal=U_prime,
                                               U_MPPI=V, state=state)

        # GET MPC OBJECTIVE
        # mppi_score_start = time()
        # Score all the rollouts
        cost_MPPI = MPPI_scores(V,state)

        weights = weight_fn(cost_MPPI)


        if jnp.isnan(cost_MPPI).any():
            print("BREAK!")
            break

        if (mppi_iter < (args.MPPI_iterations - 1)):  # and (jnp.sum(cost_MPPI*weights) < best_cost):

            # number of radars x horizon x 2
            U_prime = U_prime + jnp.sum(
                weights.reshape(args.num_traj, 1, 1) * E.reshape(args.num_traj, horizon, a_dim),
                axis=0)

            lw_cov, shrinkage = ledoit_wolf(X=E[weights != 0], assume_centered=True)

            cov_prime = jnp.array(lw_cov)

            if mppi_iter == 0:
                # print("Oracle Approx Shrinkage: ",np.round(shrinkage,5))
                pass

    if jnp.isnan(cost_MPPI).any():
        raise ValueError("Cost is NaN")

    weights = weight_fn(cost_MPPI)

    mean_shift = (U_prime - U)

    E_prime = E + mean_shift.ravel()


    U += jnp.sum(weights.reshape(-1, 1, 1) * E_prime.reshape(args.num_traj, horizon, a_dim), axis=0)

    # U = jnp.stack((jnp.clip(U[:, :, 0], control_constraints[0, 0], control_constraints[1, 0]),
    #                jnp.clip(U[:, :, 1], control_constraints[0, 1], control_constraints[1, 1])), axis=-1)
    U = jnp.stack([jnp.clip(U[...,0],control_constraints[i,0],control_constraints[i,1]) for i in range(U.shape[-1])],axis=-1)

    states = kinematic_model(U,state)


    U = jnp.roll(U, -1, axis=1)

    return U,(states,states_MPPI),cost_MPPI,key


def MPPI_visualize(MPPI_trajectories,nominal_trajectory):
    # J_eval = Multi_FIM_Logdet(U, chis, ps, qs, dts=dts, J=J, A=A, Q=Q, W=W, **key_args)
    fig,axes = plt.subplots(1,1)
    num_traj,N,time_steps,d = MPPI_trajectories.shape
    for n in range(N):
        axes.plot(MPPI_trajectories[:, n, :, 0].T, MPPI_trajectories[:, n, :, 1].T,'b-')

    axes.plot(nominal_trajectory[:, :, 0].T, nominal_trajectory[:, :, 1].T, 'r-')
    axes.axis("equal")
    plt.show()
