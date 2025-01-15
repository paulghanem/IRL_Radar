import gym
import random
import numpy as np
import torch
from gymnax.environments import EnvState
import jax
import jax.numpy as jnp

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

def generate_demo(env, env_params, model, model_params, max_frames=200,seed=123):
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

        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )

        action_seq.append(action)
        reward_seq.append(reward)

        # print(t_counter, obs, reward, action, done)
        print(f"t: {t_counter}, State: {obs}, Action: {action}, Reward: {reward}, Done: {done}")
        print(10 * "=")
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
        return np.stack(state_seq, axis=0), np.stack(action_seq, axis=0).reshape(-1,1), np.cumsum(reward_seq)
    else:
        return np.stack(state_seq,axis=0), np.stack(action_seq,axis=0), np.cumsum(reward_seq)