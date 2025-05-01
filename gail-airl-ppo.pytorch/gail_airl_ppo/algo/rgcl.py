import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.func import functional_call, hessian,jacrev

from .ppo import PPO
from gail_airl_ppo.network import RGCLCost
import torch._dynamo
torch._dynamo.config.suppress_errors = True

class RGCL(PPO):

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=10000, mix_buffer=1,
                 batch_size=1, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-5,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc=(100, 100), units_disc_v=(100, 100),
                 epoch_ppo=5, epoch_disc=1, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=10.0):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.cost = RGCLCost(
            state_shape=state_shape,
            action_shape=action_shape,
            gamma=gamma,
            hidden_units=units_disc,
            hidden_activation=nn.ReLU(inplace=True)
        ).to(device)
        
        self.params = dict(self.cost.named_parameters())
        self.param_count = sum(p.numel() for p in self.cost.parameters())
        self.param_shape = {n:p.ravel().shape[0] for n,p in self.cost.named_parameters()}
        self.theta=torch.nn.utils.parameters_to_vector(self.cost.parameters()).reshape(-1,1)

        
        #self.P_theta=1e-4*torch.eye(self.param_count)
        #self.Q_theta=1e-5*torch.eye(self.param_count)
        self.P_theta=3e-5*torch.ones(self.param_count)
        self.Q_theta=1e-6*torch.ones(self.param_count)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.cost.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc
        
        self.jac = jacrev(self.compute_cost, argnums=(0))
        self.hess = hessian(self.compute_cost, argnums=(0))
        
    def compute_cost(self, params, inputs):
        outputs = functional_call(self.cost,params, inputs)
        return outputs



    
    def update(self, writer,step=None):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, actions, _, dones, log_pis, next_states = \
                self.buffer.get_sample((step-1)%self.rollout_length)
            # Samples from expert's demonstrations.
            states_exp, actions_exp, _, dones_exp, next_states_exp = \
                self.buffer_exp.get_sample(step-1)
            # Calculate log probabilities of expert actions.
            
            # Update discriminator.
            self.update_loss(
                states, dones ,actions, next_states, states_exp,
                dones_exp , actions_exp,next_states_exp, writer
            )

        # We don't use reward signals here,
        states, actions, _, dones, log_pis, next_states = self.buffer.get()
        #states, actions, _, dones, log_pis, next_states = self.buffer.sample(self.batch_size)
        

        # Calculate rewards.
        rewards = self.cost.calculate_reward(
            states)

        # Update PPO using estimated rewards.
        
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_loss(self, states, dones,actions, next_states,
                    states_exp, dones_exp,actions_exp,
                    next_states_exp, writer):
        
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.cost(states)
        logits_exp = self.cost(
            states_exp)
        
        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        gradient_s=self.get_gradient(states,actions)
        gradient_exp=self.get_gradient(states_exp,actions_exp)
        
        #hessian_s=self.get_hessian(states,actions)
        #hessian_exp=self.get_hessian(states_exp,actions_exp)
        #hessian_s=torch.matmul(gradient_s,gradient_s.T)
        #hessian_exp=torch.matmul(gradient_exp,gradient_exp.T)
        hessian_s=torch.mul(gradient_s,gradient_s)
        hessian_exp=torch.mul(gradient_exp,gradient_exp)
        
        #self.P_theta=torch.linalg.inv(torch.linalg.inv(self.P_theta +self.Q_theta) + hessian_s - hessian_exp)
        self.P_theta=1/(1/(self.P_theta +self.Q_theta) - hessian_s + hessian_exp)
        #print(gradient_d)
        #print(gradient_s)
       
        #self.theta=self.theta-torch.matmul(self.P_theta,gradient_s-gradient_exp)
        self.theta=self.theta-torch.mul(self.P_theta,-gradient_exp+gradient_s)
        torch.nn.utils.vector_to_parameters(self.theta.ravel().detach(),self.cost.parameters())
        self.params  = dict(self.cost.named_parameters())
        
        loss = logits_exp-logits_pi
        #print(loss)

        

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss, self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)
                
  
    def get_gradient(self, states,actions):
        dc_dtheta_dict = self.jac(self.params, states)


        dc_dtheta = torch.zeros((self.param_count,1)).to(self.device)

        row = 0
        for i,(param_i_key,param_i_grad) in enumerate(dc_dtheta_dict.items()):
            row_add = self.param_shape[param_i_key]

            grad = param_i_grad.reshape(self.param_shape[param_i_key],1)

            dc_dtheta[row:(row+row_add),:] = grad

            row += row_add

        del dc_dtheta_dict
        return dc_dtheta.detach()
        
    def get_hessian(self,states,actions):
        d2c_d2theta_dict = self.hess(self.params, states)

        d2c_d2theta = torch.zeros((self.param_count,self.param_count)).to(self.device)

        row = 0
        for i,(param_i_key,param_i_hess) in enumerate(d2c_d2theta_dict.items()):
            row_add = self.param_shape[param_i_key]
            col = 0
            for j,(param_j_key, param_j_hess) in enumerate(param_i_hess.items()):
                col_add =  self.param_shape[param_j_key]

                hess_block = param_j_hess.reshape(self.param_shape[param_i_key], self.param_shape[param_j_key])

                # print(param_i_key,param_j_key,param_j_hess.shape,hess_block.shape)
                # print(torch.abs(param_j_hess.squeeze()-hess_block.squeeze()).max())
                # print(row,":",row+row_add,",",col,":",col+col_add)

                d2c_d2theta[row:(row+row_add),col:(col+col_add)] = hess_block
                col += col_add

            row += row_add

        del d2c_d2theta_dict
        # assert not (torch.abs(d2c_d2theta.transpose(1, 2) - d2c_d2theta).max() < 1e-6)
        return d2c_d2theta.detach()              
        
        
        
      