

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pdb

import os.path as osp
from pathlib import Path

gymenvs=['CartPole-v1','MountainCarContinuous-v0','HalfCHeetah-v4','Hopper','Walker2d']
#gymenvs=['CartPole-v1']
i=0
fig, ax1 = plt.subplots(1, 4, figsize=(15, 5)) 
means_table={}
std_table={}
results_final={}
for env in gymenvs:
    if env=='CartPole-v1'or env=='MountainCarContinuous-v0':
        prefix = 'cost_200'
        prefix_expert='expert_cost_200'
    else:
        prefix = 'cost_900'
        prefix_expert='expert_cost_900'
    base_dir=Path('C:/Users/siliconsynapse/Desktop/IRL_Radar/results_final')

    result_dir = osp.join(base_dir,env)
    
    if env=='Walker2d':
        methods=['gcl','rgcl','airl','gail']

    else:
        methods=['gcl','rgcl','airl','gail','sqil','UB']
    results={}
    
    for method in methods:
        folder = Path(osp.join(result_dir,method))
        files = list(folder.glob(f'{prefix}*.npy'))
        
        first_array = np.load(files[0]).reshape((-1, 1))
        results[method + '_cost'] = first_array  
        episode_length=first_array.shape[0]
        j=3
        for file in files:
            if env== 'Walker2d' and method in ['gcl','airl','gail'] :
                    continue
            
            results[method+'_cost']= np.concatenate((results[method+'_cost'],np.load(file).reshape((-1,1))),axis=1)
            j+=1
        files = list(folder.glob(f'{prefix_expert}*.npy'))
        first_array = np.load(files[0])[-1][-1]*np.ones((episode_length,1))
        results[method + '_expert_cost'] = first_array 
        for file in files:  
            if env== 'Walker2d' and method in ['gcl','airl','gail'] :
                    continue
            results[method+'_expert_cost']= np.concatenate(( results[method+'_expert_cost'],np.load(file)[-1][-1]*np.ones((episode_length,1))),axis=1)
    
    mean={}
    std={}
    expert={}
    print(env)
    if env=="Walker2d":
        for method in methods: 
            mean[method]=np.mean(results[method+'_cost']/results[method+'_expert_cost'],axis=1)
            std[method]=np.std(results[method+'_cost']/results[method+'_expert_cost'],axis=1)
            expert[method]=np.mean(results[method+'_expert_cost']/results[method+'_expert_cost'],axis=1)
    else:
        for method in methods: 
            mean[method]=np.mean(results[method+'_cost'],axis=1)
            std[method]=np.std(results[method+'_cost'],axis=1)
            expert[method]=np.mean(results[method+'_expert_cost'],axis=1)
    results_final[env+'_mean'] =mean   
    results_final[env+'_std'] =std  
    results_final[env+'_expert'] =expert
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
    for method in methods:
        means_table[method+'_'+env]=np.mean(mean[method])
        std_table[method+'_'+env]=np.mean(std[method])



# Cre79ate figure and first axis
gymenvs=['CartPole-v1','MountainCarContinuous-v0','HalfCHeetah-v4','Hopper']
for env in gymenvs:
    mean= results_final[env+'_mean']
    std=results_final[env+'_std']
    expert= results_final[env+'_expert'] 
    if env=='CartPole-v1'or env=='MountainCarContinuous-v0':
        x=np.linspace(1,21,21)
    else:
        x=np.linspace(1,91,91)
        
    if env=='Walker2d':
        methods=['gcl','rgcl','airl','gail']

    else:
        methods=['gcl','rgcl','airl','gail','sqil','UB']
    for method in methods:
        label=method
        if method=='rgcl':
            label='RDIRL'
        if method=='airl':
            label='AIRL'
        if method=='gcl':
            label='GCL'
        if method=='gail':
            label='GAIL'
        if method=='UB':
            label='UB'
        if method=='sqil':
            label='SQIL'
    
        ax1[i].plot( x,mean[method],label=label)
        if method=='gail':
            ax1[i].plot( x,expert[method],label='Expert')
        ax1[i].fill_between(x, mean[method]-std[method],mean[method]+std[method],alpha=0.1)
        ax1[i].set_xlabel('Episodes',fontsize=13)
        ax1[i].grid(True, which='both', axis='both', linestyle='--', color='gray', alpha=0.5)

        # Format x-axis dates
        fig.autofmt_xdate()
    ax1[i].set_title(env,fontsize=13)
        #ax1[i].set_xticks([1,2,3,4,5,6,7,8,9,10]) 
    i=i+1
#plt.xticks([5])
y_label = fig.supylabel('Reward', fontsize=16, x=0.07) 
plt.subplots_adjust(wspace=0.25,hspace=.5)
handles, labels = ax1[0].get_legend_handles_labels()
legend=fig.legend(handles,labels,bbox_to_anchor=(0.5, -0.05),loc='lower center',ncol=7,fontsize=14)
plt.subplots_adjust(wspace=0.4, hspace=3) 
plt.savefig('per_episode_reward_IRL_mujoco.pdf',bbox_extra_artists=(legend,y_label),bbox_inches='tight')
plt.show()


gymenvs=['Walker2d']
i=0
fig, ax1 = plt.subplots()  # One subplot
for env in gymenvs:
    mean= results_final[env+'_mean']
    std=results_final[env+'_std']
    expert= results_final[env+'_expert'] 
    if env=='CartPole-v1'or env=='MountainCarContinuous-v0':
        x=np.linspace(1,21,21)
    else:
        x=np.linspace(1,91,91)
        
    if env=='Walker2d':
        methods=['gcl','rgcl','airl','gail']

    else:
        methods=['gcl','rgcl','airl','gail','sqil','UB']
    for method in methods:
        label=method
        if method=='rgcl':
            label='RDIRL'
        if method=='airl':
            label='AIRL'
        if method=='gcl':
            label='GCL'
        if method=='gail':
            label='GAIL'
        if method=='UB':
            label='UB'
        if method=='sqil':
            label='SQIL'
    
        ax1.plot( x,mean[method],label=label)
        if method=='gail':
            ax1.plot( x,expert[method],label='Expert')
        ax1.fill_between(x, mean[method]-std[method],mean[method]+std[method],alpha=0.1)
        ax1.set_xlabel('Episodes',fontsize=13)
        ax1.grid(True, which='both', axis='both', linestyle='--', color='gray', alpha=0.5)

    
        # Format x-axis dates
        fig.autofmt_xdate()
    ax1.set_title(env,fontsize=14)
        #ax1[i].set_xticks([1,2,3,4,5,6,7,8,9,10]) 
    i=i+1
#plt.xticks([5])
y_label=fig.supylabel('Reward',fontsize=16)
plt.subplots_adjust(wspace=0.25,hspace=.5)
handles, labels = ax1.get_legend_handles_labels()
legend=fig.legend(handles,labels,bbox_to_anchor=(0.5, -0.05),loc='lower center',ncol=5,fontsize=12)

plt.savefig('per_episode_reward_IRL_Walker2d.pdf',bbox_extra_artists=(legend,y_label),bbox_inches='tight')
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



