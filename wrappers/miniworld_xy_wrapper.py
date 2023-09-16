import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
    
class MiniWorldXYWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        **kwargs
    ):
        
        gym.Wrapper.__init__(self, env)
      
        # Set-up observation space
        self.observation_space = Box(
                        low = np.array([0, env.unwrapped.min_x,env.unwrapped.min_z]),
                        high = np.array([2*np.pi,env.unwrapped.max_x, env.unwrapped.max_z]), #order??
                        **kwargs)
        
        

    def observation(self, obs):
        obs = np.array([self.env.unwrapped.agent.dir,
                           self.env.unwrapped.agent.pos[0],
                           self.env.unwrapped.agent.pos[2],
                           ])
        return obs
        


