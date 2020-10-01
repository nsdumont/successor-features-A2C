import gym
# delenvs = []
# for env in gym.envs.registry.env_specs:
#       if 'MiniGrid' in env:
#           #print("Remove {} from registry".format(env))
#           delenvs.append(env)
# for env in delenvs:
#     del gym.envs.registry.env_specs[env]
import gym_minigrid


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env.seed(seed)
    return env
