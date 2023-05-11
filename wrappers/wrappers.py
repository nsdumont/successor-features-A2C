import math
import operator
from functools import reduce

import numpy as np
import gymnasium as gym
from gymnasium import error, spaces, utils
from minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX, IDX_TO_OBJECT, IDX_TO_COLOR#, IDX_TO_STATE
import nengo_spa as spa
from gymnasium.spaces import Space
# from nengo_spa.algebras.hrr_algebra import HrrAlgebra
# from nengo.dists import Distribution, UniformHypersphere
from scipy.integrate import dblquad
#from ssp_grid_cell_utils import *

from .sspspace import *

    
class SSPWrapper(gym.core.ObservationWrapper):

    def __init__(self, env, d=193, n_scales=8, n_rotates=4,  hex=True, length_scale=2, rng=None):
        super().__init__(env)
        
        domain_bounds = np.array([[-1, env.unwrapped.width+1],[0-1, env.unwrapped.height+1],[-1,4]])
        if hex:
            self.ssp_space = HexagonalSSPSpace(3, ssp_dim=d,
                                   domain_bounds=None, scale_min=0.5, 
                                   scale_max = 2, n_scales=n_scales, n_rotates=n_rotates,
                                   length_scale=length_scale)
        else:
            self.ssp_space = RandomSSPSpace(3, ssp_dim = d,
                                   domain_bounds=None, 
                                    length_scale=length_scale,rng=rng)

        img_shape = env.observation_space['image'].shape
        self.img_shape = img_shape
        d = self.ssp_space.ssp_dim
        
        colors = [x.upper() for x in list(COLOR_TO_IDX.keys())]
        vocab = spa.Vocabulary(d, pointer_gen=np.random.RandomState(1))
        vocab.add('NULL', np.zeros(d))
        vocab.populate(';'.join(colors))
        
        objects = [x.upper() for x in list(OBJECT_TO_IDX.keys())]
        vocab.populate(';'.join(objects))
        
        #states = [x.upper() for x in list(STATE_TO_IDX.keys())]
        #vocab.populate(';'.join(states))
        vocab.add('OPEN',  vocab.algebra.identity_element(d))
        vocab.populate('CLOSED;LOCKED')
        self.vocab = vocab
            

        self.observation_space.spaces["image"] = SSPSpace(self.ssp_space)
        
        xs = [np.arange(domain_bounds[i,1],domain_bounds[i,0],-1) for i in range(3)]
        xx = np.meshgrid(*xs)
        positions = np.vstack([xx[i].reshape(-1) for i in range(3)]).T
        S_grid = self.ssp_space.encode(positions)
        S_grid = S_grid.reshape(len(xs[0]),len(xs[1]),len(xs[2]),d)#.swapaxes(0,1)
        self.S_grid = S_grid

    def observation(self, obs):
        ssp = self.ssp_space.encode(np.array([[self.env.unwrapped.agent_pos[0],
                                               self.env.unwrapped.agent_pos[1],
                                               self.env.unwrapped.agent_dir]]))
        
        # M = spa.SemanticPointer(data=np.zeros(self.d))
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         obj = img[i, j, 0]
        #         color = img[i, j, 1]
        #         state = img[i, j, 2]
        #         if obj not in [0,1]:                    
        #             S = self.S_grid[i,j,:]
                    
        #             M = M + ( S * self.vocab[IDX_TO_OBJECT[obj].upper()] * self.vocab[IDX_TO_COLOR[color].upper()] * self.vocab[IDX_TO_STATE[state].upper()])
        # #M = M.normalized()

        return {
            'mission': obs['mission'],
            'image': ssp.reshape(-1)
        }




class SSPSpace(Space):
    def __init__(self, ssp_space):
        self.ssp_space = ssp_space
        self._shape = (ssp_space.ssp_dim,)
        self.bounds = ssp_space.domain_bounds
        self.seed()

    def sample(self):
        return self.ssp_space.get_sample_ssps(1)
    
    def samples(self,n):
        return self.ssp_space.get_sample_ssps(n)
        
    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        # need more to assure its a real SSP - ie on right torus
        return (len(x) == self._shape[0])
