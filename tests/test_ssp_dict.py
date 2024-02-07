import sys,os
sys.path.insert(1, os.path.dirname(os.getcwd()))
from spaces import SSPBox, SSPDiscrete, SSPSequence, SSPDict
from wrappers import SSPEnvWrapper

import numpy as np
import gymnasium as gym

from matplotlib import animation
import matplotlib.pyplot as plt

              
ssp_dim = 151

dict_space = SSPDict({
    "object": SSPDiscrete(6, shape_out=ssp_dim),
    "position": SSPBox(-10, 10, 2, shape_out=ssp_dim, length_scale=0.1,
                          decoder_method='from-set'),
    "velocity": SSPBox(-1, 1, 2, shape_out=ssp_dim, length_scale=0.1,
                          decoder_method='from-set')
      },
    static_spaces = {"slots": SSPDiscrete(3, shape_out=ssp_dim)},
    seed=0)

print(dict_space.sample())

def map_to_dict(x):
    return {'object': x[0], 'position': x[1:3], 'velocity': x[3:]}

def map_from_dict(x_dict):
    x = np.zeros(5)
    x[0] = x_dict['object']
    x[1:3] = x_dict['position']
    x[3:] = x_dict['velocity']
    return x

def encode(x, static_spaces):
    ssp = (x['object'] * static_spaces['slots'].encode(0)   +
           x['position'] * static_spaces['slots'].encode(1)   + 
           x['velocity'] * static_spaces['slots'].encode(2)  )
    return ssp.v

def decode(ssp, spaces, static_spaces):
    x = {}
    bind = static_spaces['slots'].ssp_space.bind
    inv_slots = static_spaces['slots'].ssp_space.inverse_vectors
    x['object'] = spaces['object'].decode(bind(inv_slots[0], ssp))
    x['position'] = spaces['position'].decode(bind(inv_slots[1], ssp))
    x['velocity'] = spaces['velocity'].decode(bind(inv_slots[2], ssp))
    return x
    
    

dict_space.set_map_to_dict(map_to_dict)
dict_space.set_map_from_dict(map_from_dict)
dict_space.set_encode(encode)
dict_space.set_decode(decode)
    
print(dict_space.sample())

ssp = dict_space.encode(np.array([1,  2.3,-5.1,  -0.4,0.1]))
print(dict_space.decode(ssp))   
