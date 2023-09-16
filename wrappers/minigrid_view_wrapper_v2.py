
import numpy as np
import gymnasium as gym
import nengo_spa as spa

from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    STATE_TO_IDX
)

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

import sys,os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from spaces import SSPBox, SSPDiscrete, SSPSequence, SSPDict

    
class SSPMiniGridViewWrapper(gym.ObservationWrapper):
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
        self.view_width = env.observation_space['image'].shape[0]
        self.view_heigth = env.observation_space['image'].shape[1]
        domain_bounds = np.array([[0, self.view_width-1],
                                  [-(self.view_heigth-1)//2, (self.view_heigth-1)//2 ]])
        self.observation_space["image"] = SSPBox(
                        low = domain_bounds[:,0],
                        high = domain_bounds[:,1],
                        shape_in = 2,
                        shape_out = shape_out,
                        ssp_space=ssp_space,
                        **kwargs)
        
        
        colors = [x.upper() for x in list(COLOR_TO_IDX.keys())]
        vocab = spa.Vocabulary(shape_out, pointer_gen=np.random.RandomState(1))
        vocab.add('NULL', np.zeros(shape_out))
        vocab.populate(';'.join(colors))
        
        objects = [x.upper() for x in list(OBJECT_TO_IDX.keys())]
        vocab.populate(';'.join(objects))
        
        #states = [x.upper() for x in list(STATE_TO_IDX.keys())]
        #vocab.populate(';'.join(states))
        vocab.add('OPEN',  vocab.algebra.identity_element(shape_out))
        vocab.populate('CLOSED;LOCKED')
        self.vocab = vocab
            
        
        xs = [np.arange(domain_bounds[i,1],domain_bounds[i,0]-1,-1) for i in range(2)]
        xx = np.meshgrid(*xs)
        positions = np.vstack([xx[i].reshape(-1) for i in range(2)]).T
        S_grid = self.observation_space["image"].encode(positions)
        S_grid = S_grid.reshape(len(xs[0]),len(xs[1]),shape_out)
        self.S_grid = S_grid

    def observation(self, obs):
        img = obs['image']
        
        agt_ssp = self.observation_space["image"].encode(np.array([[self.env.unwrapped.agent_pos[0],
                                                           self.env.unwrapped.agent_pos[1],
                                                           ]]))
        
        M = np.zeros(self.observation_space["image"].shape_out)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                obj = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]
                if obj not in [0,1]:                    
                    S = self.S_grid[i,j,:]
                    obj_vec = ( self.vocab[IDX_TO_OBJECT[obj].upper()] * self.vocab[IDX_TO_COLOR[color].upper()] * self.vocab[IDX_TO_STATE[state].upper()]).v
                    M = M + self.observation_space["image"].ssp_space.bind(S, obj_vec)
        M = M / np.linalg.norm(M) 

        return {
            'mission': obs['mission'],
            'image': np.hstack([agt_ssp.reshape(-1), M.reshape(-1)])
        }


