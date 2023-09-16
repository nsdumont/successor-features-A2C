import gymnasium as gym
from gymnasium.spaces import Dict


def make_env(env_key, seed=None, **kwargs):
    env = gym.make(env_key, **kwargs) 
    env.reset(seed=seed)
    return env

