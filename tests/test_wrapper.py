import sys,os
sys.path.insert(1, os.path.dirname(os.getcwd()))
from spaces import SSPBox, SSPDiscrete, SSPSequence, SSPDict
from wrappers import SSPEnvWrapper

import numpy as np
import gymnasium as gym

from matplotlib import animation
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode='rgb_array')
env.observation_space.low = np.maximum(env.observation_space.low, -20)
env.observation_space.high = np.minimum(env.observation_space.low, 20)
env = SSPEnvWrapper(env, auto_convert_spaces = True, shape_out = 251, decoder_method = 'from-set')

# ssp_obs,_ = env.reset()
# obs = env.observation_space.decode(ssp_obs)
# ssp_action = env.action_space.sample()
# new_ssp_obs, reward, terminated, truncated, info = env.step(ssp_action)
# action = info['action']
# new_obs = info['obs']


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

observation, _ = env.reset()
frames = []
for t in range(1000):
    frames.append(env.render())
    action = env.action_space.sample()
    _, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
env.close()
save_frames_as_gif(frames)
