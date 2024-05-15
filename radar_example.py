# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:33:04 2024

@author: siliconsynapse
"""
import jax 
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import scipy.io as sio

from experts.PG import PG
from cost import CostNN
from utils import to_one_hot, get_cumulative_rewards

from torch.optim.lr_scheduler import StepLR
#torch.autograd.set_detect_anomaly(True)
# SEEDS
seed = 18095048
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
key = jax.random.PRNGKey(seed)

# ENV SETUP

n_actions = 2
M,dm=4,6
state_shape =((M*dm + 2,))

# LOADING EXPERT/DEMO SAMPLES
demo_trajs_dict =sio.loadmat('expert_samples/traj_follow.mat')
demo_trajs=demo_trajs_dict['traj']
state_0=demo_trajs[0][:-2]

print(len(demo_trajs))
states=demo_trajs[:,:-2]
actions=demo_trajs[:,-2:]

demo_trajs=[[states,actions,actions]]


# INITILIZING POLICY AND REWARD FUNCTION
policy = PG(state_shape, n_actions)
cost_f = CostNN(state_shape[0] + n_actions)
policy_optimizer = torch.optim.Adam(policy.parameters(), 1e-2)
cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-2, weight_decay=1e-4)

mean_rewards = []
mean_costs = []
mean_loss_rew = []
EPISODES_TO_PLAY = 1
REWARD_FUNCTION_UPDATE = 10
DEMO_BATCH = 100
sample_trajs = []

D_demo, D_samp = np.array([]), np.array([])

c = 299792458
fc = 1e9;
Gt = 2000;
Gr = 2000;
lam = c / fc
rcs = 1;
L = 1;
alpha = (np.pi)**2 / 3
B = 0.05 * 10**5
Pt = 10000
N=1


#

def JU_FIM_D_Radar(ps,q,Pt,Gt,Gr,L,lam,rcs,c,B,alpha):
    N,dn= ps.shape
    _,dm = q.shape
    
    
    #ps = np.concatenate((ps,np.zeros((N,1))),-1)
    q = q[:,:dm//2]

    # Qinv = jnp.linalg.inv(Q+jnp.eye(dm)*1e-8)
    # # Qinv = jnp.linalg.inv(Q)
    #
    # D11 = A.T @ Qinv @ A
    # D12 = -A.T @ Qinv
    # D21 = D12.T

    d = ps- q
    distances=np.linalg.norm(d)


    K = Pt * Gt * Gr * lam**2 * rcs / (4*np.pi)**3 / L
    C = c**2 / (alpha*B**2) * 1/K
    
    coef = (1/(C*distances**6) + 8/(distances**4))
    cost=0
    for i in range (d.shape[0]):
        cost+=np.matmul(d[i,:],d[i,:].T)
   
    
    # append zeros because there is no velocity in the radar equation...

    # zeros = jnp.zeros_like(d)
    # d = jnp.concatenate((d,zeros),-1)

    # dd^T / ||d||^4 * rho * rho/(rho+1)
    # jnp.einsum("ijk,ilm->ikm", d, d)

    # D22 = jnp.sum(outer_product,axis=0) + Qinv

    # J = D22 - D21 @ jnp.linalg.inv(J + D11) @ D12
    J= cost*coef

    return J

def JU_FIM_radareqn_target_logdet(ps,qs,
                               Pt,Gt,Gr,L,lam,rcs,c,B,alpha):

    # FIM of single target, multiple sensors

    FIM = JU_FIM_D_Radar(ps=ps,q=qs,
                         Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,c=c,B=B,alpha=alpha)

    # sign,logdet = jnp.linalg.slogdet(jnp.linalg.inv(FIM)+jnp.eye(FIM.shape[0])*1e-5)
    # logdet = -logdet
    logdet = np.log(FIM)
    return logdet


# CONVERTS TRAJ LIST TO STEP LIST
def preprocess_traj(traj_list, step_list, is_Demo = False):
    step_list = step_list.tolist()
    for traj in traj_list:
        states = np.array(traj[0])
        if is_Demo:
            probs = np.ones((states.shape[0], 1))
        else:
            probs = np.array(traj[1]).reshape(-1, 1)
        actions = np.array(traj[2])
        x = np.concatenate((states, probs, actions), axis=1)
        step_list.extend(x)
    return np.array(step_list)

D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
return_list, sum_of_cost_list = [], []

chis = jax.random.uniform(key,shape=(N,1),minval=-np.pi,maxval=np.pi)
for i in range(1000):
    trajs = [policy.generate_session_radar(state_0,chis,M,dm) for _ in range(EPISODES_TO_PLAY)]
    sample_trajs = trajs + sample_trajs
    D_samp = preprocess_traj(trajs, D_samp)

    # UPDATING REWARD FUNCTION (TAKES IN D_samp, D_demo)
    loss_rew = []
    for _ in range(REWARD_FUNCTION_UPDATE):
        selected_samp = np.random.choice(len(D_samp), DEMO_BATCH)
        selected_demo = np.random.choice(len(D_demo), DEMO_BATCH)

        D_s_samp = D_samp[selected_samp]
        D_s_demo = D_demo[selected_demo]

        #D̂ samp ← D̂ demo ∪ D̂ samp
        D_s_samp = np.concatenate((D_s_demo, D_s_samp), axis = 0)

        states, probs, actions = D_s_samp[:,:-3], D_s_samp[:,-3], D_s_samp[:,-2:]
        states_expert, actions_expert = D_s_demo[:,:-3], D_s_demo[:,-2:]

        # Reducing from float64 to float32 for making computaton faster
        states = torch.tensor(states, dtype=torch.float32)
        probs = torch.tensor(probs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        states_expert = torch.tensor(states_expert, dtype=torch.float32)
        actions_expert = torch.tensor(actions_expert, dtype=torch.float32)

        costs_samp = cost_f(torch.cat((states, actions), dim=-1))
        costs_demo = cost_f(torch.cat((states_expert, actions_expert), dim=-1))
       
        # LOSS CALCULATION FOR IOC (COST FUNCTION)
        loss_IOC = torch.mean(costs_demo) + \
                torch.log(torch.mean(torch.exp(-costs_samp)/(probs+1e-7)))
        # UPDATING THE COST FUNCTION
        cost_optimizer.zero_grad()
        loss_IOC.backward()
        cost_optimizer.step()

        loss_rew.append(loss_IOC.detach())

    for traj in trajs:
        states, probs ,actions= traj
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        
            
        costs = cost_f(torch.cat((states, actions), dim=-1))
        cumulative_returns = torch.tensor(get_cumulative_rewards(-costs, 0.99))
        #cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)
        
        
        probs=policy(states)
        log_probs = torch.log(probs)

    
        entropy = -torch.mean(torch.sum(probs*log_probs), dim = -1 )
        loss = torch.mean(costs-1e-2*log_probs.reshape(-1,1))-entropy*1e-2 
        #loss = -torch.mean(log_probs*cumulative_returns -entropy*1e-2)

        # UPDATING THE POLICY NETWORK
        policy_optimizer.zero_grad()
        loss.backward()
        policy_optimizer.step()

    
    sum_of_cost = np.sum(costs.detach().numpy())
    
    sum_of_cost_list.append(sum_of_cost)

    
    mean_costs.append(np.mean(sum_of_cost_list))
    mean_loss_rew.append(np.mean(loss_rew))

    # PLOTTING PERFORMANCE
    if i % 10 == 0:
        
        costs_GT=0
        # for j in range(M):
        #     ps=states_expert[:,:3]
        #     qs=states_expert[:,3+(j*dm):3+(j+1)*dm]
            
        #     costs_GT+= JU_FIM_radareqn_target_logdet(ps,qs,
        #                                Pt,Gt,Gr,L,lam,rcs,c,B,alpha)
        # clear_output(True)
        print(f"mean reward:{np.mean(return_list)} loss: {loss_IOC}")
        print(f"learned cost:{np.sum(costs_demo.detach().numpy())} GT: {costs_GT} Loss_policy: {loss} loss_IOC: {loss_IOC}")
        plt.subplot(2, 2, 1)
        plt.title(f"Mean cost per {EPISODES_TO_PLAY} games")
        plt.plot(mean_costs)
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.title(f"Mean loss per {REWARD_FUNCTION_UPDATE} batches")
        plt.plot(mean_loss_rew)
        plt.grid()

        # plt.show()
        plt.savefig('plots/GCL_learning_curve.png')
        plt.close()

    if np.mean(return_list) > 500:
        break
