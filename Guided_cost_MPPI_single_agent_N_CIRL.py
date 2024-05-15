# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:38:21 2024

@author: siliconsynapse
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 00:00:04 2024

@author: siliconsynapse
"""


import jax
import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import scipy.io as sio
from copy import deepcopy
from jax import config
from flax.training import train_state,checkpoints
import flax 
import optax





from experts.P_MPPI import *
from cost_jax import CostNN, apply_model, update_model
from utils import to_one_hot, get_cumulative_rewards

from torch.optim.lr_scheduler import StepLR
from src_range.FIM_new.FIM_RADAR import Single_JU_FIM_Radar,Single_FIM_Radar,Multi_FIM_Logdet_decorator_MPC,FIM_Visualization
from src_range.FIM_new.generate_demo_fn import generate_demo,generate_demo_MPC
from src_range.utils import NoiseParams
from src_range.control.MPPI import *



# CONVERTS TRAJ LIST TO STEP LIST
def preprocess_traj(traj_list, step_list, is_Demo = False):
    step_list = step_list.tolist()
    for traj in traj_list:
        states = jnp.array(traj[0])
        if is_Demo:
            probs = jnp.ones((states.shape[0], 1))
        else:
            probs = jnp.array(traj[1]).reshape(-1, 1)
        actions = jnp.array(traj[2])
        x = jnp.concatenate((states, probs, actions), axis=1)
        step_list.extend(x)
    return jnp.array(step_list)


#torch.autograd.set_detect_anomaly(True)
# SEEDS
seed = 123
key = jax.random.PRNGKey(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ENV SETUP

n_actions = 2
M,dm=1,6
N,dn=1,2
state_shape =((M*(dm//2) + N*(dn+1),))

# LOADING EXPERT/DEMO SAMPLES
demo_trajs_dict =sio.loadmat('expert_samples/Single_traj_follow.mat')
demo_trajs=demo_trajs_dict['traj']
input_shape=demo_trajs[0].flatten().shape[0]-2+1

print(len(demo_trajs))
states=demo_trajs[:,:-2]
actions=demo_trajs[:,-2:]

#states=states.reshape((states.shape[0],states.shape[1]*states.shape[2]))#watch out for the order
#actions=actions.reshape((actions.shape[0],actions.shape[1]*actions.shape[2]))# watch out for the order
    

demo_trajs=[[states,actions,actions]]


# INITILIZING POLICY AND REWARD FUNCTION
policy = P_MPPI(state_shape, n_actions)
cost_f = CostNN(state_dims=state_shape[0])
#cost_optimizer = torch.optim.Adam(cost_f.parameters(), 1e-2, weight_decay=1e-4)
init_rng = jax.random.key(0)

variables = cost_f.init(init_rng, jnp.ones((1,input_shape))) 

params = variables['params']
tx = optax.adam(learning_rate=1e-2)
state_train=train_state.TrainState.create(apply_fn=cost_f.apply, params=params, tx=tx)

mean_rewards = []
mean_costs = []
mean_loss_rew = []
EPISODES_TO_PLAY = 1
REWARD_FUNCTION_UPDATE = 10
DEMO_BATCH = 100
sample_trajs = []

D_demo, D_samp = np.array([]), jnp.array([])

c = 299792458
fc = 1e9;
Gt = 2000;
Gr = 2000;
lam = c / fc
rcs = 1;
L = 1;
alpha = (jnp.pi)**2 / 3
B = 0.05 * 10**5
# calculate Pt such that I achieve SNR=x at distance R=y
R = 1000

Pt = 10000
K = Pt * Gt * Gr * lam ** 2 * rcs / L / (4 * jnp.pi) ** 3
Pr = K / (R ** 4)

# get the power of the noise of the signalf
SNR=0
N = 1
T = .05
NT = 300

ps = jax.random.uniform(key, shape=(N, 2), minval=-100, maxval=100)


ps_init = deepcopy(ps)
chis = jax.random.uniform(key,shape=(ps.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))
z_elevation = 10
qs = jnp.array([[0.0, -0.0,z_elevation, 25., 20,0]])
m0=qs
pt=qs[:,:2]
chit = jax.random.uniform(key,shape=(pt.shape[0],1),minval=-jnp.pi,maxval=jnp.pi) #jnp.tile(0., (ps.shape[0], 1, 1))


sigmaQ = jnp.sqrt(10 ** -1)
sigmaV = jnp.sqrt(9)

A_single = jnp.array([[1., 0, 0, T, 0, 0],
               [0, 1., 0, 0, T, 0],
               [0, 0, 1, 0, 0, T],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 1., 0],
               [0, 0, 0, 0, 0, 1]])

Q_single = jnp.array([
    [(T ** 4) / 4, 0, 0, (T ** 3) / 2, 0, 0],
    [0, (T ** 4) / 4, 0, 0, (T ** 3) / 2, 0],
    [0, 0, (T**4)/4, 0, 0, (T**3) / 2],
    [(T ** 3) / 2, 0, 0, (T ** 2), 0, 0],
    [0, (T ** 3) / 2, 0, 0, (T ** 2), 0],
    [0, 0, (T**3) / 2, 0, 0, (T**2)]
]) * sigmaQ ** 2

A = jnp.kron(jnp.eye(M), A_single);
Q = jnp.kron(jnp.eye(M), Q_single);
G = jnp.eye(N)

nx = Q.shape[0]

J = jnp.eye(dm*M) #jnp.stack([jnp.eye(d) for m in range(M)])

time_steps = 20
time_step_size = T
max_velocity = 50.
min_velocity = 0
max_angle_velocity = jnp.pi
min_angle_velocity = -jnp.pi
M, dm = qs.shape;
N , dn = ps.shape;
method = "Single_FIM_3D_action_MPPI_NN"
method_t = "Single_FIM_3D_action_MPPI_NN_t"
#MPPI_method = "single"
MPPI_method = "NN"
MPPI_method_t="NN_t"
sigmaW = jnp.sqrt(M*Pr/ (10**(SNR/10)))
v_init = 0
av_init = 0
U_V = jnp.ones((N,time_steps,1)) * v_init
U_W = jnp.ones((N,time_steps,1)) * av_init
U_Nom =jnp.concatenate((U_V,U_W),axis=-1)

# ==================== MPPI CONFIGURATION ================================= #
limits = jnp.array([[max_velocity, max_angle_velocity], [min_velocity, min_angle_velocity]])
Qinv = jnp.linalg.inv(Q+jnp.eye(dm*M)*1e-8)

IM_fn = partial(Single_JU_FIM_Radar,A=A,Qinv=Qinv,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaV=sigmaV,sigmaW=sigmaW,method=MPPI_method)
IM_fn_t = partial(Single_JU_FIM_Radar,A=A,Qinv=Qinv,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaV=sigmaV,sigmaW=sigmaW,method=MPPI_method_t)
#IM_fn(ps,qs,J=J,actions=U_Nom)
IM_fn_GT = partial(Single_JU_FIM_Radar,A=A,Qinv=Qinv,Pt=Pt,Gt=Gt,Gr=Gr,L=L,lam=lam,rcs=rcs,fc=fc,c=c,sigmaV=sigmaV,sigmaW=sigmaW)

Multi_FIM_Logdet = Multi_FIM_Logdet_decorator_MPC(IM_fn=IM_fn,method=method)
Multi_FIM_Logdet_t = Multi_FIM_Logdet_decorator_MPC(IM_fn=IM_fn_t,method=method_t)
MPPI_scores = MPPI_scores_wrapper(Multi_FIM_Logdet,method=MPPI_method)
MPPI_scores_t = MPPI_scores_wrapper(Multi_FIM_Logdet_t,method=MPPI_method)

#D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
#D_demo=jnp.concatenate((D_demo[:,:2],jnp.zeros((D_demo.shape[0],1)),D_demo[:,2:]),axis=1)
return_list, sum_of_cost_list = [], []

#U,chis,radar_states,target_states


for i in range(100):
    if (i % 5 == 0): 
    
        demo_trajs=generate_demo(state_train)
        demo_trajs=np.array(demo_trajs)
    
        states_d=demo_trajs[:,:-2]
        actions_d=demo_trajs[:,-2:]
        D_demo=np.array([])
    
        demo_trajs=[[states_d,actions_d,actions_d]]
        D_demo = preprocess_traj(demo_trajs, D_demo, is_Demo=True)
        D_demo=jnp.concatenate((D_demo[:,:2],jnp.zeros((D_demo.shape[0],1)),D_demo[:,2:]),axis=1)
    
    trajs = [policy.generate_session_N_CIRL(NT,ps,chis, pt,chit,m0,A,time_steps,time_step_size,limits,MPPI_scores,MPPI_scores_t,state_train,i,IM_fn_GT) for _ in range(EPISODES_TO_PLAY)]
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
        D_s_samp = jnp.concatenate((D_s_demo, D_s_samp), axis = 0)

        states, probs, actions = D_s_samp[:,:-3], D_s_samp[:,-3], D_s_samp[:,-2:]
        states_expert, actions_expert = D_s_demo[:,:-3], D_s_demo[:,-2:]

        # Reducing from float64 to float32 for making computaton faster
        #states = torch.tensor(states, dtype=torch.float32)
        #probs = torch.tensor(probs, dtype=torch.float32)
        #actions = torch.tensor(actions, dtype=torch.float32)
        #states_expert = torch.tensor(states_expert, dtype=torch.float32)
        #actions_expert = torch.tensor(actions_expert, dtype=torch.float32)
        grads, loss_IOC = apply_model(state_train, states, actions,states_expert,actions_expert,probs)
        state_train = update_model(state_train, grads)
        
       
     
        # UPDATING THE COST FUNCTION
       # cost_optimizer.zero_grad()
        #loss_IOC.backward()
        #cost_optimizer.step()

        loss_rew.append(loss_IOC)

    # for traj in trajs:
    #     states, probs ,actions= traj
        
    #     states = torch.tensor(states, dtype=torch.float32)
    #     actions = torch.tensor(actions, dtype=torch.float32)
        
            
    #     costs = cost_f(torch.cat((states, actions), dim=-1))
    #     cumulative_returns = torch.tensor(get_cumulative_rewards(-costs, 0.99))
    #     #cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)
        
        
    #     probs=policy(states)
    #     log_probs = torch.log(probs)

    
    #     entropy = -torch.mean(torch.sum(probs*log_probs), dim = -1 )
    #     loss = torch.mean(costs-1e-2*log_probs.reshape(-1,1))-entropy*1e-2 
    #     #loss = -torch.mean(log_probs*cumulative_returns -entropy*1e-2)

    #     # UPDATING THE POLICY NETWORK
        
        
        

    
    # sum_of_cost = np.sum(costs.detach().numpy())
    
    # sum_of_cost_list.append(sum_of_cost)

    
    # mean_costs.append(np.mean(sum_of_cost_list))
    mean_loss_rew.append(np.mean(loss_rew))
    
    

    # PLOTTING PERFORMANCE
    if i % 10 == 0:
        
        costs_GT=0
        # for j in range(M):
        #     ps=states_expert[:,:3]
        #     qs=states_expert[:,3+(j*dm):3+(j+1)*dm]
            
        #     costs_GT+= JU_FIM_radareqn_target_logdet(ps,qs,
        #                                Pt,Gt,Gr,L,lam,rcs,c,B,alpha)
        # # clear_output(True)
        print(f"mean loss:{np.mean(loss_rew)} loss: {loss_IOC} epoch: {i}")
        # print(f"learned cost:{np.sum(costs_demo.detach().numpy())} GT: {costs_GT} Loss_policy: {loss} loss_IOC: {loss_IOC}")
        # plt.subplot(2, 2, 1)
        # plt.title(f"Mean cost per {EPISODES_TO_PLAY} games")
        # plt.plot(mean_costs)
        # plt.grid()

        plt.subplot(2, 2, 2)
        plt.title(f"Mean loss per {REWARD_FUNCTION_UPDATE} batches")
        plt.plot(mean_loss_rew)
        plt.grid()

        # plt.show()
        plt.savefig('plots/GCL_learning_curve.png')
        plt.close()

    if np.mean(return_list) > 500:
        break
    
config = {'dimensions': np.array([5, 3])}
ckpt_single = {'model_single': state_train, 'config': config, 'data': [D_samp]}
checkpoints.save_checkpoint(ckpt_dir='/tmp/flax_ckpt/flax-checkpointing',
                            target=ckpt_single,
                            step=0,
                            overwrite=True,
                            keep=2)