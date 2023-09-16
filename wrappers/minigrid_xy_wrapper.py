import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
    
class MiniGridXYWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        **kwargs
    ):
        
        gym.Wrapper.__init__(self, env)
     
        # Set-up observation space
        self.observation_space["image"] = Box(
                        low = np.array([0,0,0]),
                        high = np.array([3,env.unwrapped.width,env.unwrapped.height]),
                        dtype=np.float32,
                        **kwargs)
        
        

    def observation(self, obs):
        xy_obs = np.array([self.env.unwrapped.agent_dir,
                        self.env.unwrapped.agent_pos[0],
                        self.env.unwrapped.agent_pos[1],
                        ])
        return {
            'mission': obs['mission'],
            'image': xy_obs
        }


