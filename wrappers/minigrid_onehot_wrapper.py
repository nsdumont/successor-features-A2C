import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
    
class MiniGridOneHotWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        **kwargs
    ):
        
        gym.Wrapper.__init__(self, env)
        
        self.dir_size = 4
        self.width_size = env.unwrapped.width
        self.height_size = env.unwrapped.height
        self.size = self.dir_size*self.width_size*self.height_size
        # Set-up observation space
        self.observation_space["image"] = Box(
                        low = np.zeros(self.size),
                        high = np.ones(self.size),
                        dtype=np.float32,
                        **kwargs)
        
        

    def observation(self, obs):
        mat_obs = np.zeros((self.dir_size,self.width_size,self.height_size))
        mat_obs[int(self.env.unwrapped.agent_dir),
                int(self.env.unwrapped.agent_pos[0]),
                int(self.env.unwrapped.agent_pos[1]) ] = 1
        
        return {
            'mission': obs['mission'],
            'image': mat_obs.reshape(-1)
        }


