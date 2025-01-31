# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:51:09 2025

@author: siliconsynapse
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pdb

import os.path as osp

gymenvs=['CartPole-v1','MountainCarContinuous-v0']
#gymenvs=['CartPole-v1']
i=0
fig, ax1 = plt.subplots(1,2)
means_table={}
std_table={}
for env in gymenvs:

    epoch_time_dir = osp.join('results','plotting',env)
    epoch_cost_dir = osp.join('results','plotting',env)
    
    methods=['gcl','rgcl','airl','gail']
    results={}
    for method in methods:
        results[method+'_epoch_time']=np.load(osp.join(epoch_time_dir,method+'_epoch_time.npy'))
        results[method+'_epoch_cost']= np.load(osp.join(epoch_cost_dir,method+'_epoch_cost.npy'))
        results[method+'_expert_cost']=np.load(osp.join(epoch_cost_dir,'gail'+'_expert_cost.npy'))    
    
    
    mean={}
    std={}
    expert={}
    for method in methods: 
        mean[method]=np.mean(results[method+'_epoch_cost'],axis=0)
        std[method]=np.std(results[method+'_epoch_cost'],axis=0)
        expert[method]=np.mean(results[method+'_expert_cost'][0][0][-1]*np.ones((6,10)),axis=0)
           
    # epoch_time_GCL=np.load('epoch_time_GCL.npy')
    # FIM_expert_GCL=np.load('FIM_expert_GCL.npy')
    # FIM_true_GCL=np.load('FIM_true_GCL.npy')
    # #FIM_predicted_GCL=np.load('FIM_predicted_GCL.npy')
    
    # Mse_gcl=np.mean(np.square(FIM_expert_GCL-FIM_true_GCL),axis=1)
    # distance_gcl=np.mean(np.abs(FIM_expert_GCL-FIM_true_GCL),axis=1)
    # #Mse_gcl_pred=np.mean(np.square(FIM_predicted_GCL-FIM_true_GCL),axis=1)
    
    # epoch_time_RGCL=np.load('epoch_time_RGCL.npy')
    # FIM_expert_RGCL=np.load('FIM_expert.npy')
    # FIM_true_RGCL=np.load('FIM_true_RGCL.npy')
    # #FIM_predicted_RGCL=np.load('FIM_predicted_RGCL.npy')
    
    # Mse_rgcl=np.mean(np.square(FIM_expert_RGCL-FIM_true_RGCL),axis=1)
    # distance_rgcl=np.mean(np.abs(FIM_expert_RGCL-FIM_true_RGCL),axis=1)
    # #Mse_rgcl_pred=np.mean(np.square(FIM_predicted_RGCL-FIM_true_RGCL),axis=1)
    
    # epoch_time_GGCL=np.load('epoch_time_GGCL.npy')
    # FIM_expert_GGCL=np.load('FIM_expert_GGCL.npy')
    # FIM_true_GGCL=np.load('FIM_true_GGCL.npy')
    
    # Mse_ggcl=np.mean(np.square(FIM_expert_GGCL-FIM_true_GGCL),axis=1)
    
    
    # epoch_time_GAIL=np.load('epoch_time_GAIL.npy')
    # FIM_expert_GAIL=np.load('FIM_expert_GAIL.npy')
    # FIM_true_GAIL=np.load('FIM_true_GAIL.npy')
    
    # Mse_gail=np.mean(np.square(FIM_expert_GAIL-FIM_true_GAIL),axis=1)
    
    
    
    #pdb.set_trace()
        means_table[method+'_'+env]=np.mean(mean[method])
        std_table[method+'_'+env]=np.std(results[method+'_epoch_cost'])
    
    
    x=np.linspace(1,10,10)
    # Create figure and first axis
    
    for method in methods:
        label=method
        if method=='rgcl':
            label='RDIRL'
        if method=='airl':
            label='GAN-GCL'
        if method=='gcl':
            label='GCL'
        if method=='gail':
            label='GAIL'
    
        ax1[i].plot( x,mean[method],label=label)
        if method=='gail':
            ax1[i].plot( x,expert[method],label='Expert')
        ax1[i].fill_between(x, mean[method]-std[method],mean[method]+std[method],alpha=0.1)
        ax1[i].set_xlabel('Episodes')
    
        # Format x-axis dates
        fig.autofmt_xdate()
        ax1[i].set_title(env)
        ax1[i].set_xticks([1,2,3,4,5,6,7,8,9,10]) 
    i=i+1
#plt.xticks([5])
y_label=fig.supylabel('Reward')
plt.subplots_adjust(wspace=0.25,hspace=.5)
handles, labels = ax1[0].get_legend_handles_labels()
legend=fig.legend(handles,labels,bbox_to_anchor=(0.5, -0.05),loc='lower center',ncol=5)

plt.savefig('per_episode_reward_IRL_gym.pdf',bbox_extra_artists=(legend,y_label),bbox_inches='tight')
plt.show()



# fig2, ax1 = plt.subplots()
# ax1.plot(Mse_gcl_pred, color='blue')
# ax1.set_ylabel('GCL', color='blue')

# ax1.plot(Mse_rgcl_pred, color='red')
# # Format x-axis dates
# fig.autofmt_xdate()

# plt.show()




# time_axis_gcl=np.cumsum(epoch_time_GCL)
# time_axis_rgcl=np.cumsum(epoch_time_RGCL)
# Create figure and first axis



