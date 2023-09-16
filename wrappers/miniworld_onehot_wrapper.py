import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
    
class MiniWorldOneHotWrapper(gym.ObservationWrapper):
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
        self.dir_size = 4
        self.min_x = env.unwrapped.min_x
        self.min_z = env.unwrapped.min_z
        
        self.width_size = env.unwrapped.max_x - env.unwrapped.min_x
        self.height_size = env.unwrapped.max_z - env.unwrapped.min_z
        self.size = self.dir_size*self.width_size*self.height_size
        # Set-up observation space
        self.observation_space["image"] = Box(
                        low = np.zeros(self.size),
                        high = np.ones(self.size),
                        dtype=np.float32,
                        **kwargs)
        
        
        

    def observation(self, obs):
        mat_obs = np.zeros((self.dir_size,self.width_size,self.height_size))
        x = self.env.unwrapped.agent.pos[0] + self.min_x
        z = self.env.unwrapped.agent.pos[1] + self.minz
        mat_obs[int(self.env.unwrapped.agent.dir),
                int(x),
                int(z) ] = 1
        
        return mat_obs.reshape(-1)


