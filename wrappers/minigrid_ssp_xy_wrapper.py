import numpy as np
import gymnasium as gym

import sys,os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from spaces import SSPBox

    
class SSPMiniGridXYWrapper(gym.ObservationWrapper):
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
        self.observation_space["image"] = SSPBox(
                        low = np.array([0,0,0]),
                        high = np.array([3,env.unwrapped.width,env.unwrapped.height]),
                        shape_in = 3,
                        shape_out = shape_out,
                        ssp_space=ssp_space,
                        **kwargs)
        
        

    def observation(self, obs):
        ssp_obs = self.observation_space["image"].encode(np.array([[self.env.unwrapped.agent_dir,
                                                           self.env.unwrapped.agent_pos[0],
                                                           self.env.unwrapped.agent_pos[1],
                                                           ]]))
        return {
            'mission': obs['mission'],
            'image': ssp_obs.reshape(-1)
        }


