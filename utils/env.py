import gymnasium as gym


def make_env(env_key, seed=None, **kwargs):
    env = gym.make(env_key,  **kwargs)
    env.reset(seed=seed)
    return env