import gymnasium as gym
import random
import numpy as np
import torch
from gymnax.environments import EnvState
import jax
import jax.numpy as jnp
import pdb

def load_config(config_fname, seed_id=0, lrate=None):
    """Load training configuration and random seed of experiment."""
    import yaml
    import re
    from dotmap import DotMap

    def load_yaml(config_fname: str) -> dict:
        """Load in YAML config file."""
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        with open(config_fname) as file:
            yaml_config = yaml.load(file, Loader=loader)
        return yaml_config

    config = load_yaml(config_fname)
    config["train_config"]["seed_id"] = seed_id
    if lrate is not None:
        if "lr_begin" in config["train_config"].keys():
            config["train_config"]["lr_begin"] = lrate
            config["train_config"]["lr_end"] = lrate
        else:
            try:
                config["train_config"]["opt_params"]["lrate_init"] = lrate
            except Exception:
                pass
    return DotMap(config)


def save_pkl_object(obj, filename):
    """Helper to store pickle objects."""
    import pickle
    from pathlib import Path

    output_file = Path(filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(filename, "wb") as output:
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    print(f"Stored data at {filename}.")


def load_pkl_object(filename: str):
    """Helper to reload pickle objects."""
    import pickle

    with open(filename, "rb") as input:
        obj = pickle.load(input)
    print(f"Loaded data from {filename}.")
    return obj

def get_cumulative_rewards(rewards, gamma=0.99):
    G = torch.zeros_like(rewards, dtype=float)
    G[-1] = rewards[-1]
    for idx in range(-2, -len(rewards)-1, -1):
        G[idx] = rewards[idx] + gamma * G[idx+1]
    return G


def to_one_hot(y_tensor, ndims):
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(
        y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot

from flax import struct
from gymnax.utils.state_translate import control_np_to_jax


import os.path as osp

from utils.models import load_neural_network
import gymnax
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np

import gymnasium as gym
from gymnasium import Wrapper


import torch
import numpy as np

def fast_predict(model, obs, device="cpu", deterministic=True):
    """
    Faster replacement for SB3's model.predict().
    - Skips VecEnv wrapping/unwrapping
    - Skips numpy <-> torch overhead
    - Returns torch tensor (no .numpy() sync)
    """
    policy = model.policy
    obs = np.array(obs)  # ensure numpy
    if obs.ndim == 1:    # add batch dimension
        obs = obs[None, :]

    obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
    with torch.no_grad():
        actions, _, _ = policy(obs_tensor, deterministic=deterministic)

    return actions.squeeze(0)  # remove batch dim



class CustomTerminationWrapper(Wrapper):
    def __init__(self, env, max_steps=200):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.current_step += 1


        # Override termination condition
        # Terminate only if max steps reached
        if self.current_step > self.max_steps:
            terminated = True
        else:
            terminated = False

        return observation, reward, terminated, truncated, info


class GenerateDemo(object):
    def __init__(self,env_name,max_frames=200):
        self.env_name = env_name
        self.base = osp.join("expert_agents", env_name)
        self.max_frames = max_frames

        # Map environments to their expert models in the experts/ directory
        self.expert_models = {
            'HalfCheetah-v4': ('experts/td3-HalfCheetah-v3.zip', TD3),
            'Hopper-v4': ('experts/td3-Hopper-v3.zip', TD3),
            'Walker2d-v4': ('experts/td3-Walker2d-v3.zip', TD3),
            'Walker2d': ('experts/td3-Walker2d-v3.zip', TD3),
            'Swimmer-v4': ('experts/td3-Swimmer-v4.zip', TD3),
            'Swimmer': ('experts/td3-Swimmer-v4.zip', TD3),
            'Hopper': ('experts/td3-Hopper-v3.zip', TD3),
        }


    def generate_demo(self,seed=123):
        # Map of all MuJoCo and Gymnasium environments that use Stable-Baselines3
        gymnasium_envs = [
            "MountainCarContinuous-v0",
            "HalfCheetah-v4",
            "Ant",
            "Ant-v4",
            "Hopper",
            "Hopper-v4",
            "Walker2d",
            "Walker2d-v4",
            "Humanoid-v4",
            "Swimmer",
            "Swimmer-v4"
        ]

        if self.env_name in gymnasium_envs:
            states,actions,rewards,env = self.generate_gymnasium_demo(self.env_name,max_frames=self.max_frames,seed=seed)

        else:
            states,actions,rewards,env = self.generate_gymnax_demo(self.env_name, max_frames=self.max_frames, seed=seed)

        return states,actions,rewards,env

    def generate_gymnasium_demo(self,env_name,max_frames=200,seed=123):
        # Enjoy trained agent

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)



        if self.env_name in ["MountainCarContinuous-v0"]:
            env = CustomTerminationWrapper(gym.make(env_name, render_mode='rgb_array'),max_steps=max_frames)

            model = DDPG("MlpPolicy", env)
            base = osp.join(self.base,f"{env_name}.zip")
        else:
            # Check if we have a downloaded expert model
            if self.env_name in self.expert_models:
                expert_path, algo_class = self.expert_models[self.env_name]
                print(f"Loading expert from: {expert_path} using {algo_class.__name__}")

                # Determine correct environment version
                if env_name.endswith('-v4'):
                    actual_env_name = env_name
                else:
                    actual_env_name = f"{env_name}-v4"

                # Create environment
                # Note: v3 experts were trained WITH exclude_current_positions_from_observation=True (default)
                # We need to match the observation space the model was trained on
                if actual_env_name.endswith('-v3'):
                    # v3 excludes x-position by default
                    env = CustomTerminationWrapper(
                        gym.make(actual_env_name),
                        max_steps=max_frames
                    )
                elif "Swimmer" in actual_env_name:
                    # Swimmer-v4 has XML compatibility issues, use v3 instead
                    env = CustomTerminationWrapper(
                        gym.make("Swimmer-v3"),
                        max_steps=max_frames
                    )
                else:
                    # v4 includes x-position by default, but experts are from v3
                    # So we need to match v3 behavior
                    env = CustomTerminationWrapper(
                        gym.make(actual_env_name, exclude_current_positions_from_observation=True),
                        max_steps=max_frames
                    )

                # Load the appropriate algorithm
                model = algo_class("MlpPolicy", env, verbose=1)
                base = expert_path
            else:
                # Fallback to old behavior for environments without downloaded experts
                if self.env_name == "Ant":
                    env = CustomTerminationWrapper(gym.make(env_name,exclude_current_positions_from_observation=False),max_steps=max_frames)
                else:
                    env = CustomTerminationWrapper(gym.make(env_name,exclude_current_positions_from_observation=False),max_steps=max_frames)

                model = PPO("MlpPolicy", env,verbose=1)
                base = osp.join(self.base,"PPO.zip")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.load(base, env,device)
       


        vec_env = model.get_env()
        vec_env._seeds = [seed]
        obs = vec_env.reset()

        # Access underlying gym environment to get full MuJoCo state
        base_env = vec_env.envs[0].unwrapped

        state_seq = []  # Observations for policy
        full_state_seq = []  # Full state (qpos + qvel) for dynamics/rewards
        action_seq = []
        reward_seq = []
        t_counter = 0

        # Initial full state
        if hasattr(base_env, 'data'):  # MuJoCo environment
            qpos = base_env.data.qpos.copy()
            qvel = base_env.data.qvel.copy()
            full_state = np.concatenate([qpos, qvel])
            full_state_seq.append(full_state)

        while True:
            # Save observation (for compatibility/debugging)
            state_seq.append(obs.ravel())

            # Predict action using observation
            #action = fast_predict(model, obs)
            with torch.no_grad():
                action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, info = vec_env.step(action)

            # Extract full state AFTER stepping
            if hasattr(base_env, 'data'):  # MuJoCo environment
                qpos = base_env.data.qpos.copy()
                qvel = base_env.data.qvel.copy()
                full_state = np.concatenate([qpos, qvel])
                full_state_seq.append(full_state)

            action_seq.append(action.ravel())
            reward_seq.append(reward)
            reward_sum=np.sum(reward_seq)

            # print(t_counter, obs, reward, action, done)
            print(f"t: {t_counter}, State: {obs}, Action: {action}, Reward: {reward},Reward_sum: {reward_sum}, Done: {done}")
            print(10 * "=")
            t_counter += 1
            if t_counter == max_frames:
                break


        print(f"{env_name} - Steps: {t_counter}, Return: {np.sum(reward_seq)}, State: {obs}")

        # Return full_state_seq instead of obs-based state_seq for MuJoCo environments
        # This ensures state includes x-position for reward calculation
        if hasattr(base_env, 'data'):
            # Use full state (includes x-position) - shape will be (T+1, full_state_dim)
            # Trim to match action length
            full_states = np.stack(full_state_seq[:-1], axis=0)  # Use states, not next_states
        else:
            # Non-MuJoCo environments use observations
            full_states = np.stack(state_seq, axis=0)

        if len(action.shape) == 0:
            return full_states, np.stack(action_seq, axis=0).reshape(-1,1), np.cumsum(reward_seq), vec_env
        else:
            return full_states, np.stack(action_seq,axis=0), np.cumsum(reward_seq), vec_env



    def generate_gymnax_demo(self,env_name, max_frames=200,seed=123):

        policy_method_agent = "es" if env_name == "MountainCarContinuous-v0" else "ppo"
        base = osp.join(self.base, policy_method_agent)
        configs = load_config(base + ".yaml")
        model, model_params = load_neural_network(
            configs.train_config, base + ".pkl"
        )

        env, env_params = gymnax.make(
            configs.train_config.env_name,
            **configs.train_config.env_kwargs,
        )
        env_params.replace(**configs.train_config.env_params)

        state_seq = []
        action_seq = []
        rng = jax.random.PRNGKey(seed)


        rng, rng_reset = jax.random.split(rng)
        obs, env_state = env.reset(rng_reset, env_params)

        t_counter = 0
        reward_seq = []
        while True:

            state_seq.append(obs)

            rng, rng_act, rng_step = jax.random.split(rng, 3)

            if model.model_name.startswith("separate"):
                v, pi = model.apply(model_params, obs, rng_act)
                action = pi.sample(seed=rng_act)
                if env.name == "CartPole-v1":
                    action = action*env_params.force_mag + (1-action)*(-env_params.force_mag)

            else:
                action = model.apply(model_params, obs, rng_act)
            #pdb.set_trace()
            next_obs, next_env_state, reward, done, info = env.step(
                rng_step, env_state, action, env_params
            )

            action_seq.append(action)
            reward_seq.append(reward)
            #reward_seq.append(reward)
            reward_sum=np.sum(reward_seq)

            # print(t_counter, obs, reward, action, done)
            #print(f"t: {t_counter}, State: {obs}, Action: {action}, Reward: {reward},Reward_sum: {reward_sum}, Done: {done}")
            #print(10 * "=")
            t_counter += 1
            if env.name == "MountainCarContinuous-v0":
                if done:
                    break
            if t_counter == max_frames:
                break
            else:
                env_state = next_env_state
                obs = next_obs
        print(f"{env.name} - Steps: {t_counter}, Return: {np.sum(reward_seq)}, State: {obs}")

        if len(action.shape) == 0:
            return np.stack(state_seq, axis=0), np.stack(action_seq, axis=0).reshape(-1,1), np.cumsum(reward_seq),env
        else:
            return np.stack(state_seq,axis=0), np.stack(action_seq,axis=0), np.cumsum(reward_seq),env