import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jax import config
from scipy.stats import multivariate_normal
import math

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import jit

class PG(nn.Module):
    def __init__(self, state_shape, n_actions):
        
        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(in_features = state_shape[0], out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128 , out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features = 64 , out_features = self.n_actions*2)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-3)
        
    def pdf(self,mean,cov,x):
        x=torch.reshape(x,(x.shape[0],1))
        mean=torch.reshape(mean,(x.shape[0],1))
        pdf=(2*math.pi)**(-1)*(torch.linalg.det(cov))**(-1/2)*torch.exp(-1/2*torch.matmul(torch.matmul((x-mean).T,torch.linalg.inv(cov)),(x-mean)))
        return pdf
        
    def forward(self, x):
        out = self.model(x)
        mean=out[:,0:2]
        var=nn.ReLU()(out[:,2:]) + 1e-1
        #var=torch.tensor([1e0,1e0])
        
        probs=torch.zeros((x.shape[0]))
        for i in range(x.shape[0]):
            cov=torch.tensor([[var[i,0],0],[0,var[i,1]]])
            #cov=torch.tensor([[var[0],0],[0,var[1]]])
            pd=torch.distributions.multivariate_normal.MultivariateNormal(mean[i,:], cov)
            a = pd.sample()
            probs[i]=self.pdf(mean[i,:],cov,a)
        return probs
    
    def predict_probs(self, states):
        states = torch.FloatTensor(states)
        out = self.model(states).detach()
        mean=out[0][0:2]
        var=nn.ReLU()(out[0][2:]) + 1e-1
        #var=np.array([1e0,1e0])
        cov=np.array([[var[0],0],[0,var[1]]])
        a =  np.random.multivariate_normal(mean, cov, 1)
        probs=multivariate_normal.pdf(a, mean=mean, cov=cov,allow_singular=True)
        #probs=probs.clip(0,1)
        #probs = F.softmax(logits, dim = -1).numpy()
        # print(states, logits, probs)
        return a, probs
    
    def state_multiple_update(self,p,U,chi,time_step_sizes):
        # sensor dynamics for unicycle model

        vs,avs = U[0][0],U[0][1]

        chi = chi.reshape(1,1)
        p = p.reshape(1,-1)

        chi_next = chi + time_step_sizes * avs
        

        ps_next = p + jnp.column_stack((jnp.cos(chi.ravel()),
                                                   jnp.sin(chi.ravel()))) * vs * time_step_sizes
       

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

        return ps_next,chi_next

    
    def generate_session(self, env, t_max=1000):
        states, traj_probs, actions, rewards = [], [], [], []
        s = env.reset()
        q_t = 1.0
        for t in range(t_max):
            action_probs = self.predict_probs(np.array([s]))[0]
            a = np.random.choice(self.n_actions,  p = action_probs)
            new_s, r, done, info = env.step(a)
            
            q_t *= action_probs[a]

            states.append(s)
            traj_probs.append(q_t)
            actions.append(a)
            rewards.append(r)

            s = new_s
            if done:
                break

        return states, traj_probs, actions
    
    def generate_session_radar(self,s,chis, M, dm, t_max=1000):
        states, traj_probs, actions, rewards = [], [], [], []
        q_t = 1.0
        T = .05
        A = jnp.array([[1., 0, 0, T, 0, 0],
                       [0, 1., 0, 0, T, 0],
                       [0, 0, 1, 0, 0, T],
                       [0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 1., 0],
                       [0, 0, 0, 0, 0, 1]])
        for t in range(t_max):
            a,prob = self.predict_probs(np.array([s]))
            ps=s[0:2]
            chi=chis
            ps_next,chi_next= self.state_multiple_update(ps,a,chi,T)
            qs=s[2:]
            qs = (A @ qs.reshape(-1, dm).T).T.reshape(M, dm)
            qs=qs.flatten()
            new_s=np.concatenate((ps_next.flatten(),qs))
            
            q_t *= prob

            states.append(s)
            traj_probs.append(q_t)
            actions.append(a.flatten())
            

            s = new_s
            chis=chi_next

        return states, traj_probs, actions

    def _get_cumulative_rewards(self, rewards, gamma=0.99):
        G = np.zeros_like(rewards, dtype = float)
        G[-1] = rewards[-1]
        for idx in range(-2, -len(rewards)-1, -1):
            G[idx] = rewards[idx] + gamma * G[idx+1]
        return G

    def _to_one_hot(self, y_tensor, ndims):
        y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
        y_one_hot = torch.zeros(
            y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
        return y_one_hot

    def train_on_env(self, env, gamma=0.99, entropy_coef=1e-2):
        states, actions, rewards = self.generate_session(env)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int32)
        cumulative_returns = np.array(self._get_cumulative_rewards(rewards, gamma))
        cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

        logits = self.model(states)
        probs = nn.functional.softmax(logits, -1)
        log_probs = nn.functional.log_softmax(logits, -1)

        log_probs_for_actions = torch.sum(
            log_probs * self._to_one_hot(actions, env.action_space.n), dim=1)
    
        entropy = -torch.mean(torch.sum(probs*log_probs), dim = -1 )
        loss = -torch.mean(log_probs_for_actions*cumulative_returns -entropy*entropy_coef)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return np.sum(rewards)