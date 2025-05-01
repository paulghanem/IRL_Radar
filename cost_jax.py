

# See issue #620.
# pytype: disable=wrong-keyword-args

from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from functools import partial
import pdb
from jax.tree_util import tree_flatten, tree_unflatten
#import tensorflow_datasets as tfds


class CostNN(nn.Module):

  state_dims: int
  hidden_dim: int
  out_features = 1

  @nn.compact
  def __call__(self, x):
    #x = nn.relu(x)
    x = nn.Dense(self.hidden_dim)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_dim)(x)
    x = nn.relu(x)
    x = nn.Dense(self.out_features)(x)
    x = jnp.clip(x**2,min=0,max=5) #
    return x


def cost_fn(state_train,params,states,N):
   
    #costs_demo = -jnp.log(state_train.apply_fn({'params': params}, states_expert)+1e-2)
    #costs_samp =-jnp.log(state_train.apply_fn({'params': params}, states)+1e-2)
    costs = (state_train.apply_fn({'params': params}, states)+1e-6).flatten()/N
    
    return costs[0].astype(float)

jax.jit
def apply_model(state_train, states, actions,states_expert,actions_expert,probs,probs_experts,UB=False):
    """Computes gradients, loss and accuracy for a single batch."""
    

  
    def loss_fn(params):

        #costs_demo = -jnp.log(state_train.apply_fn({'params': params}, states_expert)+1e-2)
        #costs_samp =-jnp.log(state_train.apply_fn({'params': params}, states)+1e-2)
        costs_demo = state_train.apply_fn({'params': params}, states_expert)+1e-6
        costs_samp = state_train.apply_fn({'params': params}, states)+1e-6
      # LOSS CALCULATION FOR IOC (COST FUNCTION)
      #logits = state_train.apply_fn({'params': params}, jnp.concatenate((states,actions),axis=1))
        # g_demo=jnp.zeros(costs_demo.shape)
        # g_mono_demo=jnp.zeros(costs_demo.shape)
        # g_samp=jnp.zeros(costs_samp.shape)
        # g_mono_samp=jnp.zeros(costs_samp.shape)
        # for i in range (2,costs_demo.shape[0]):
        #     g_demo=g_demo.at[i].set(jnp.pow((costs_demo[i]-costs_demo[i-1])-(costs_demo[i-1]-costs_demo[i-2]),2))
        #     g_mono_demo=g_mono_demo.at[i].set(jnp.pow(jnp.maximum(0,costs_demo[i]-costs_demo[i-1]-1),2))
        # for i in range (costs_samp.shape[0]):
        #     g_samp=g_samp.at[i].set(jnp.pow((costs_samp[i]-costs_samp[i-1])-(costs_samp[i-1]-costs_samp[i-2]),2))
        #     g_mono_samp=g_mono_samp.at[i].set(jnp.pow(jnp.maximum(0,costs_samp[i]-costs_samp[i-1]-1),2))  
         
       # loss = jnp.mean(costs_demo) + \
         #      jnp.log(jnp.mean(jnp.exp(-costs_samp)/(probs+1e-7))) #+ jnp.sum(g_demo) + jnp.sum(g_samp)+jnp.sum(g_mono_demo)+jnp.sum(g_mono_samp)
               #jnp.mean((jnp.exp(-costs_demo))/(probs_experts))
        #loss=jnp.mean(optax.l2_loss(predictions=costs_samp,targets=jnp.ones((200,1))))
        if UB:
            loss = jnp.mean(costs_demo) + \
                   jnp.mean(-costs_samp)
        else: 
            loss = jnp.mean(costs_demo) + \
                   jnp.log(jnp.mean(jnp.exp(-costs_samp)/(probs+1e-7)))
       
            
        # loss = jnp.mean(((costs_demo) + \
        #        jnp.log(jnp.exp(-costs_samp)/(probs+1e-7)))**2)
   
        return loss
      
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state_train.params)
    return grads, loss


@jax.jit
def apply_model_AIRL(state_train, states, actions,states_expert,actions_expert,probs,probs_experts,UB=False):
    """Computes gradients, loss and accuracy for a single batch."""
    

  
    def loss_fn(params):
 
        costs_demo = state_train.apply_fn({'params': params}, states_expert)+1e-6
        costs_samp = state_train.apply_fn({'params': params}, states)+1e-6
        
        disc_demo = jnp.divide(jnp.exp(-costs_demo),(jnp.exp(-costs_demo)+1))
        disc_samp = jnp.divide(jnp.exp(-costs_samp),(jnp.exp(-costs_samp)+1))
        

      # LOSS CALCULATION FOR IOC (COST FUNCTION)
      #logits = state_train.apply_fn({'params': params}, jnp.concatenate((states,actions),axis=1))
        # g_demo=jnp.zeros(costs_demo.shape)
        # g_mono_demo=jnp.zeros(costs_demo.shape)
        # g_samp=jnp.zeros(costs_samp.shape)
        # g_mono_samp=jnp.zeros(costs_samp.shape)
        # for i in range (2,costs_demo.shape[0]):
        #     g_demo=g_demo.at[i].set(jnp.pow((costs_demo[i]-costs_demo[i-1])-(costs_demo[i-1]-costs_demo[i-2]),2))
        #     g_mono_demo=g_mono_demo.at[i].set(jnp.pow(jnp.maximum(0,costs_demo[i]-costs_demo[i-1]-1),2))
        # for i in range (costs_samp.shape[0]):
        #     g_samp=g_samp.at[i].set(jnp.pow((costs_samp[i]-costs_samp[i-1])-(costs_samp[i-1]-costs_samp[i-2]),2))
        #     g_mono_samp=g_mono_samp.at[i].set(jnp.pow(jnp.maximum(0,costs_samp[i]-costs_samp[i-1]-1),2))  
         
        loss = -jnp.mean(jnp.log(disc_demo+1e-6))-jnp.mean(jnp.log(1-disc_samp+1e-6))
              
        return loss
      
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state_train.params)
    return grads, loss

@jax.jit
def apply_model_multi(state_train, states, actions,states_expert,actions_expert,probs,N):
    """Computes gradients, loss and accuracy for a single batch."""
    
   
  
    def loss_fn(params):
        costs_demo=0
        costs_samp=0
        N,dn=5,2
        for j in range (N):
            ps=states_expert[:,j:j+2]
            qs=states_expert[:,N*dn:]
            combined_expert=jnp.concatenate((ps,qs),axis=1)
            ps=states[:,j:j+2]
            qs=states[:,N*dn:]
            combined=jnp.concatenate((ps,qs),axis=1)
            
            costs_demo+=jnp.log(state_train.apply_fn({'params': params}, jnp.concatenate((combined_expert,actions_expert[:,j:j+2]),axis=1)))
            costs_samp+=jnp.log(state_train.apply_fn({'params': params}, jnp.concatenate((combined,actions[:,j:j+2]),axis=1)))
        
        
      # LOSS CALCULATION FOR IOC (COST FUNCTION)
      #logits = state_train.apply_fn({'params': params}, jnp.concatenate((states,actions),axis=1))
        loss = jnp.mean(costs_demo) + \
              jnp.log(jnp.mean(jnp.exp(-costs_samp)/(probs+1e-7)))
      #loss=jnp.mean(optax.l2_loss(predictions=costs_samp,targets=jnp.ones((200,1))))
        
        return loss
      
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state_train.params)
    return grads, loss

    
@jax.jit 
def get_gradients(state_train,params,state,N_steps):
    #pdb.set_trace()
    dc_d_theta=jax.grad(cost_fn,argnums=1)(state_train,params,state.reshape(1,-1),N_steps)
    # Dense_0_bias_g=dc_d_theta['Dense_0']['bias']
    # Dense_0_kernel_g=dc_d_theta['Dense_0']['kernel'].reshape((dc_d_theta['Dense_0']['kernel'].shape[0]*dc_d_theta['Dense_0']['kernel'].shape[1]))
    # Dense_1_bias_g=dc_d_theta['Dense_1']['bias']
    # Dense_1_kernel_g=dc_d_theta['Dense_1']['kernel'].reshape((dc_d_theta['Dense_1']['kernel'].shape[0]*dc_d_theta['Dense_1']['kernel'].shape[1]))
    
    # gradients=jnp.concatenate((Dense_0_bias_g,Dense_0_kernel_g,Dense_1_bias_g,Dense_1_kernel_g))
    flat_grads, tree_def = tree_flatten(dc_d_theta)
    gradients=jnp.concatenate([p.flatten() for p in flat_grads])
    return gradients

@jax.jit 
def get_hessian(state_train,params,state,N_steps):
    d2c_d2_theta=jax.hessian(cost_fn,argnums=1)(state_train,params,state.reshape(1,-1),N_steps)
    Dense_0_bias_h=jax.tree.flatten(d2c_d2_theta['Dense_0']['bias'])
    Dense_0_kernel_h=jax.tree.flatten(d2c_d2_theta['Dense_0']['kernel'])
    Dense_1_bias_h=jax.tree.flatten(d2c_d2_theta['Dense_1']['bias'])
    Dense_1_kernel_h=jax.tree.flatten(d2c_d2_theta['Dense_1']['kernel'])
    Dense_2_bias_h=jax.tree.flatten(d2c_d2_theta['Dense_2']['bias'])
    Dense_2_kernel_h=jax.tree.flatten(d2c_d2_theta['Dense_2']['kernel'])
    
    for j in range(len(Dense_0_bias_h[0])):
        Dense_0_bias_h[0][j]=Dense_0_bias_h[0][j].reshape((Dense_0_bias_h[0][j].shape[0],-1))
        Dense_1_bias_h[0][j]=Dense_1_bias_h[0][j].reshape((Dense_1_bias_h[0][j].shape[0],-1))
        Dense_2_bias_h[0][j]=Dense_2_bias_h[0][j].reshape((Dense_2_bias_h[0][j].shape[0],-1))
        Dense_0_kernel_h[0][j]=Dense_0_kernel_h[0][j].reshape((Dense_0_kernel_h[0][j].shape[0]*Dense_0_kernel_h[0][j].shape[1],-1))
        Dense_1_kernel_h[0][j]=Dense_1_kernel_h[0][j].reshape((Dense_1_kernel_h[0][j].shape[0]*Dense_1_kernel_h[0][j].shape[1],-1))
        Dense_2_kernel_h[0][j]=Dense_2_kernel_h[0][j].reshape((Dense_2_kernel_h[0][j].shape[0]*Dense_2_kernel_h[0][j].shape[1],-1))
    
    hessian=jnp.concatenate((jnp.concatenate(Dense_0_bias_h[0],axis=1),jnp.concatenate(Dense_0_kernel_h[0],axis=1),jnp.concatenate(Dense_1_bias_h[0],axis=1),jnp.concatenate(Dense_1_kernel_h[0],axis=1),jnp.concatenate(Dense_2_bias_h[0],axis=1),jnp.concatenate(Dense_2_kernel_h[0],axis=1)),axis=0)
    #flat_hessian, tree_def = tree_flatten(d2c_d2_theta)
   # hessian=jnp.concatenate([p.flatten() for p in flat_hessian])
    return hessian

@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)
  

def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size
      
    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
      
    epoch_loss = []
    epoch_accuracy = []
      
    for perm in perms:
      batch_images = train_ds['image'][perm, ...]
      batch_labels = train_ds['label'][perm, ...]
      grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
      state = update_model(state, grads)
      epoch_loss.append(loss)
      epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy
  
  
def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.0
    return train_ds, test_ds
      
      
def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)
      
  
def train_and_evaluate(
      config: ml_collections.ConfigDict, workdir: str
      ) -> train_state.TrainState:
    """Execute model training and evaluation loop.
      
    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
      
    Returns:
      The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets()
    rng = jax.random.key(0)
      
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))
      
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)
      
    for epoch in range(1, config.num_epochs + 1):
      rng, input_rng = jax.random.split(rng)
      state, train_loss, train_accuracy = train_epoch(
          state, train_ds, config.batch_size, input_rng
      )
      _, test_loss, test_accuracy = apply_model(
          state, test_ds['image'], test_ds['label']
      )
      
      logging.info(
          'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f,'
          ' test_accuracy: %.2f'
          % (
              epoch,
              train_loss,
              train_accuracy * 100,
              test_loss,
              test_accuracy * 100,
          )
      )
      
      summary_writer.scalar('train_loss', train_loss, epoch)
      summary_writer.scalar('train_accuracy', train_accuracy, epoch)
      summary_writer.scalar('test_loss', test_loss, epoch)
      summary_writer.scalar('test_accuracy', test_accuracy, epoch)
      
    summary_writer.flush()
    return state
