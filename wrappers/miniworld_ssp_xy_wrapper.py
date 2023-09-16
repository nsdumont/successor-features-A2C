import numpy as np
import gymnasium as gym

import sys,os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from spaces import SSPBox

    
class SSPMiniWorldXYWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        ssp_space = None,
        shape_out = None,
        **kwargs
    ):
        
        gym.Wrapper.__init__(self, env)
     
        if shape_out is not None:
            assert(type(shape_out) is int), f"Expects `shape_out` to be an integer, actual type: {type(shape_out)}"
            
        # Set-up observation space
        self.observation_space = SSPBox(
                        low = np.array([0, env.unwrapped.min_x,env.unwrapped.min_z]),
                        high = np.array([2*np.pi,env.unwrapped.max_x, env.unwrapped.max_z]), #order??
                        shape_in = 3,
                        shape_out = shape_out,
                        ssp_space=ssp_space,
                        **kwargs)
        
        

    def observation(self, obs):
        ssp_obs = self.observation_space.encode(np.array([[self.env.unwrapped.agent.dir,
                                                           self.env.unwrapped.agent.pos[0],
                                                           self.env.unwrapped.agent.pos[2],
                                                           ]]))
        return ssp_obs.reshape(-1)
        


