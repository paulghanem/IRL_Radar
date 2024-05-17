# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNIST example.

Library file which executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

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
#import tensorflow_datasets as tfds


class CostNN(nn.Module):
 
  state_dims: int
  hidden_dim1 = 128
  out_features = 1

  @nn.compact
  def __call__(self, x):  
    #x = nn.relu(x)
    x = nn.Dense(self.hidden_dim1)(x)
    x = nn.relu(x)
    x = nn.Dense(self.out_features)(x)
    x = nn.relu(x)
    return x


@jax.jit
def apply_model(state_train, states, actions,states_expert,actions_expert,probs,probs_experts):
    """Computes gradients, loss and accuracy for a single batch."""
    

  
    def loss_fn(params):
        costs_demo = -jnp.log(state_train.apply_fn({'params': params}, states_expert)+1e-2)
        costs_samp =-jnp.log(state_train.apply_fn({'params': params}, states)+1e-2)
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
         
        loss = jnp.mean(costs_demo) + \
               jnp.log(jnp.mean(jnp.exp(-costs_samp)/(probs+1e-7))) #+ jnp.sum(g_demo) + jnp.sum(g_samp)+jnp.sum(g_mono_demo)+jnp.sum(g_mono_samp)
               #jnp.mean((jnp.exp(-costs_demo))/(probs_experts))
        #loss=jnp.mean(optax.l2_loss(predictions=costs_samp,targets=jnp.ones((200,1))))
          
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
