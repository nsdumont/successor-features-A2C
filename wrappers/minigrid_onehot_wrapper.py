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


class MazeOneHotWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        **kwargs
    ):
        
        gym.Wrapper.__init__(self, env)

        self.width_size = env.unwrapped.maze_view.maze.MAZE_W
        self.height_size = env.unwrapped.maze_view.maze.MAZE_H
        self.size = self.width_size*self.height_size
        # Set-up observation space
        self.observation_space = Box(
                        low = np.zeros(self.size),
                        high = np.ones(self.size),
                        dtype=np.float32,
                        **kwargs)
        
        

    def observation(self, obs):
        hot_obs = np.zeros((self.width_size,self.height_size))
        hot_obs[int(obs[0]), int(obs[1])] = 1.0
        
        return hot_obs.reshape(-1)
        

