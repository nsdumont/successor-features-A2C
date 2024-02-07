import sys,os
sys.path.insert(1, os.path.dirname(os.getcwd()))
from spaces import SSPBox, SSPDiscrete, SSPSequence, SSPDict
import numpy as np

ssp_dim = 151

box_space = SSPBox(-1,1, 2, shape_out=ssp_dim, decoder_method='from-set')
print(box_space.decode(box_space.encode(np.array([0.1,-0.3]))))

discrete_space = SSPDiscrete(3, shape_out=ssp_dim)
print(discrete_space.decode(discrete_space.encode(1)))

seq_space = SSPSequence(SSPBox(-1,1, 2, shape_out=ssp_dim, decoder_method='from-set', length_scale=0.1), length = 3)
seq = np.array([[0.1,-0.3],[0,-0.1],[-0.2,0.5]])
print(seq_space.decode(seq_space.encode(seq.reshape(-1))))

dict_space = SSPDict({"speed": SSPBox(-1,1, 2, shape_out=ssp_dim, decoder_method='from-set'), 
      "color": SSPDiscrete(3, shape_out=ssp_dim),
      "position": SSPSequence(SSPBox(-10,10, 2, shape_out=ssp_dim, decoder_method='from-set'), 5)}, seed=0)
print(dict_space.sample())

