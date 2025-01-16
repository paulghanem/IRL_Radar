# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:51:09 2025

@author: siliconsynapse
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



epoch_time_GCL=np.load('epoch_time_GCL.npy')
FIM_expert_GCL=np.load('FIM_expert_GCL.npy')
FIM_true_GCL=np.load('FIM_true_GCL.npy')
#FIM_predicted_GCL=np.load('FIM_predicted_GCL.npy')

Mse_gcl=np.mean(np.square(FIM_expert_GCL-FIM_true_GCL),axis=1)
distance_gcl=np.mean(np.abs(FIM_expert_GCL-FIM_true_GCL),axis=1)
#Mse_gcl_pred=np.mean(np.square(FIM_predicted_GCL-FIM_true_GCL),axis=1)

epoch_time_RGCL=np.load('epoch_time_RGCL.npy')
FIM_expert_RGCL=np.load('FIM_expert.npy')
FIM_true_RGCL=np.load('FIM_true_RGCL.npy')
#FIM_predicted_RGCL=np.load('FIM_predicted_RGCL.npy')

Mse_rgcl=np.mean(np.square(FIM_expert_RGCL-FIM_true_RGCL),axis=1)
distance_rgcl=np.mean(np.abs(FIM_expert_RGCL-FIM_true_RGCL),axis=1)
#Mse_rgcl_pred=np.mean(np.square(FIM_predicted_RGCL-FIM_true_RGCL),axis=1)

epoch_time_GGCL=np.load('epoch_time_GGCL.npy')
FIM_expert_GGCL=np.load('FIM_expert_GGCL.npy')
FIM_true_GGCL=np.load('FIM_true_GGCL.npy')

Mse_ggcl=np.mean(np.square(FIM_expert_GGCL-FIM_true_GGCL),axis=1)


epoch_time_GAIL=np.load('epoch_time_GAIL.npy')
FIM_expert_GAIL=np.load('FIM_expert_GAIL.npy')
FIM_true_GAIL=np.load('FIM_true_GAIL.npy')

Mse_gail=np.mean(np.square(FIM_expert_GAIL-FIM_true_GAIL),axis=1)
# Create figure and first axis
fig, ax1 = plt.subplots()
ax1.plot( Mse_gcl[:], color='blue',label='GCL')
ax1.set_xlabel('Env runs')
ax1.plot( Mse_rgcl[:60], color='red',label='RDIRL(ours)')
ax1.plot( Mse_ggcl[:],label='GAN-GCL')
ax1.plot( Mse_gail[:],label='GAIL')
# Format x-axis dates
fig.autofmt_xdate()
plt.title('Cognitive radar')
plt.xticks([0, 50, 100,150,200,250,300,350,400,450,500]) 
legend=plt.legend(bbox_to_anchor=(0.5, -0.4),loc='lower center',ncol=4)

plt.savefig('per_epoch_mse_IRL_radar.pdf',bbox_extra_artists=(legend,), bbox_inches='tight')
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



