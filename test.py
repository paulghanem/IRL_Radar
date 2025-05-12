

from __future__ import annotations
from flax.training import train_state,checkpoints
import flax 
import optax
import os
import argparse
import os.path as osp

import numpy as np
import jax.numpy as jnp
import jax
from jax import vmap,jit
import time 

import gymnax
import gymnasium as gym
import mujoco
from gymnax.visualize import Visualizer
from flax import struct
from gymnax.environments import EnvState
import pdb
from mujoco import mjx 

# from experts.P_MPPI import P_MPPI
from cost_jax import CostNN, apply_model, apply_model_AIRL,update_model

from src.objective_fns.cost_to_go_fns import get_cost
from src.control.dynamics import get_state
from src.control.mppi_class import MPPI
from src.control.dynamics import get_action_cov,get_action_space,get_step_model

from utils.helpers import GenerateDemo

import gymnax

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

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import jit

from flax import struct
import chex
from typing import Any, Dict, Optional, Tuple, Union
from jax import lax
from gymnax.environments import environment
from gymnax.environments import spaces
import pdb
from mujoco import mjx 
from functools import partial



from typing import Callable, Tuple, Dict
import jax
import jax.numpy as jnp
from jax.random import multivariate_normal

from src.control.dynamics import kinematics,kinematics_mujoco
from src.objective_fns.cost_to_go_fns import get_cost
from cost_jax import get_gradients,get_hessian

import os.path as osp

import numpy as np
import math
from tqdm.auto import tqdm
from copy import deepcopy
#from utils.models import load_neural_network
import gymnax
import gymnasium as gym
import pdb
from cost_jax import apply_model, apply_model_AIRL, update_model
from mujoco import mjx 


import os
import gymnasium as gym
from stable_baselines3 import PPO,SAC

import gymnasium as gym
import random
import numpy as np
import torch
from gymnax.environments import EnvState
import jax
import jax.numpy as jnp
import pdb

import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import jax
import jax.numpy as jnp
import flax.linen as nn
from tensorflow_probability.substrates import jax as tfp
from evosax import NetworkMapper
import gymnax
from utils.helpers import load_pkl_object

from functools import partial
import optax
import jax
import jax.numpy as jnp
from typing import Any, Callable, Tuple
from collections import defaultdict
import flax
from flax.training.train_state import TrainState
import numpy as np
import tqdm
import gymnax