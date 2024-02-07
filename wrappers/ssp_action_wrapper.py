import numpy as np
import gymnasium as gym
import sys,os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from spaces import SSPBox, SSPDiscrete

class SSPMiniActionWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        seed = None,
        shape_out = None,
        **kwargs
    ):
        
        gym.Wrapper.__init__(self, env)
     
        if shape_out is not None:
            assert(type(shape_out) is int), f"Expects `shape_out` to be an integer, actual type: {type(shape_out)}"
            
        # Set-up action space
        self.action_space = SSPDiscrete(
                        n = env.action_space.n,
                        shape_out = shape_out,
                        seed=seed,
                        **kwargs)
        
        

    def action(self, action):
        ssp_action = self.action_space.encode(np.array([[action]]))
        return ssp_action.reshape(-1)
        
