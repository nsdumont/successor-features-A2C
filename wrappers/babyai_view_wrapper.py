
import numpy as np
import gymnasium as gym
import nengo_spa as spa
import re

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

    
class SSPBabyAIViewWrapper(gym.ObservationWrapper):
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
        domain_bounds = np.array([ [0, self.view_width-1],
                                  [-(self.view_heigth-1)//2, (self.view_heigth-1)//2 ],
                                  [0,3]])
        self.observation_space["image"] = SSPBox(
                        low = domain_bounds[:,0],
                        high = domain_bounds[:,1],
                        shape_in = 3,
                        shape_out = shape_out,
                        ssp_space=ssp_space,
                        **kwargs)
        
        self.ssp_dim = self.observation_space["image"].shape_out
        self.observation_space["image"]._shape = 2*self.ssp_dim
        
        self.vocab = spa.Vocabulary(self.ssp_dim, pointer_gen=np.random.RandomState(1))
        self.vocab.add('I', self.vocab.algebra.identity_element(self.ssp_dim))
        self.vocab.add('NULL', self.vocab.algebra.zero_element(self.ssp_dim))
        self.vocab.populate('HAS_KEY;HAS_BALL;GO_TO;PUT;NEXT_TO;OPEN;PICK_UP' )
        
        colors = [x.upper() for x in list(COLOR_TO_IDX.keys())]
        self.color_map =  dict(zip(list(COLOR_TO_IDX.keys()), colors))
        self.vocab.populate(';'.join(colors))
        
        objects = [x for x in list(OBJECT_TO_IDX.keys())]
        notice_obj = ['NULL', 'NULL', 'NULL', 'NULL', 'DOOR', 'KEY', 'BALL', 'BOX', 'GOAL', 'LAVA', 'NULL']
        self.obj_map = dict(zip(objects, notice_obj))
        self.vocab.populate(';'.join([o for o in notice_obj if o not in self.vocab.keys()]))
        
        states = [x for x in list(STATE_TO_IDX.keys())]
        notice_states = ['I', 'CLOSED', 'LOCKED']
        self.state_map = dict(zip(states, notice_states))
        self.vocab.populate(';'.join([o for o in notice_states if o not in self.vocab.keys()]))
                    
        self.total_map = self.color_map | self.obj_map | self.state_map
        
        xs = [np.arange(domain_bounds[i,1],domain_bounds[i,0]-1,-1) for i in range(2)]
        xx = np.meshgrid(*xs)
        positions = np.array([3,6]).reshape(1,-1) - np.vstack([xx[i].reshape(-1) for i in range(2)]).T
        positions = np.hstack([positions,-1*np.ones((positions.shape[0],1))])
        S_grid = self.observation_space["image"].encode(positions)
        S_grid = S_grid.reshape(len(xs[0]),len(xs[1]),self.ssp_dim)
        self.S_grid = S_grid
        
        

    def observation(self, obs):
        img = obs['image']
        
        agt_ssp = self.observation_space["image"].encode(np.array([[self.env.unwrapped.agent_pos[0],
                                                           self.env.unwrapped.agent_pos[1],
                                                           self.env.unwrapped.agent_dir
                                                           ]]))
        # self.observation_space["image"].encode(np.array([[self.env.unwrapped.agent_pos[0],
        #                                                    self.env.unwrapped.agent_pos[1],
        #                                                    ]]))
        if self.env.unwrapped.carrying is not None:## change this!!
            agt_ssp = self.observation_space["image"].ssp_space.bind(agt_ssp, self.vocab['HAS_KEY'].v)
        #M = np.zeros(self.observation_space["image"].shape_out)
        M = agt_ssp
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                obj = img[i, j, 0]
                color = img[i, j, 1]
                state = img[i, j, 2]
                if obj not in [0,1]:     
                    obj_ssp = self.S_grid[i,j,:]
                    obj_name = self.obj_map[IDX_TO_OBJECT[obj]]  
                    col_name =  self.color_map[IDX_TO_COLOR[color]]
                    state_name = self.state_map[IDX_TO_STATE[state]]
                    
                    obj_sp = ( self.vocab[obj_name] * self.vocab[col_name] * self.vocab[state_name]).v
                    M = M + self.observation_space["image"].ssp_space.bind(obj_ssp, obj_sp)

        M = M / np.linalg.norm(M) 
        
        mission = obs['mission']
        O = np.zeros(M.shape)
        if re.match("go to the .*", mission):
            words = mission.partition("go to the ")[2].split(" ")
            O_w = self.vocab['GO_TO']
            for w in words:
                O_w = O_w * self.vocab[self.total_map[w]]
            O += O_w.v
        if re.match("get a .*", mission): #same as to go to
            words = mission.partition("get a ")[2].split(" ")
            O_w = self.vocab['GO_TO']
            for w in words:
                O_w = O_w * self.vocab[self.total_map[w]]
            O += O_w.v
        if re.match("go get a .*", mission):#same as to go to
            words = mission.partition("go get a ")[2].split(" ")
            O_w = self.vocab['GO_TO']
            for w in words:
                O_w = O_w * self.vocab[self.total_map[w]]
            O += O_w.v
        if re.match("fetch a .*", mission):#same as to go to
            words = mission.partition("fetch a ")[2].split(" ")
            O_w = self.vocab['GO_TO']
            for w in words:
                O_w = O_w * self.vocab[self.total_map[w]]
            O += O_w.v
        if re.match("go fetch a .*", mission):#same as to go to
            words = mission.partition("go fetch a ")[2].split(" ")
            O_w = self.vocab['GO_TO']
            for w in words:
                O_w = O_w * self.vocab[self.total_map[w]]
            O += O_w.v
        if re.match("you must fetch a .*", mission):#same as to go to
            words = mission.partition("you must fetch a ")[2].split(" ")
            O_w = self.vocab['GO_TO']
            for w in words:
                O_w = O_w * self.vocab[self.total_map[w]]
            O += O_w.v
        if re.match(".*open the .*", mission):
            words = mission.partition("open the ")[2].split(" ")
            O_w = self.vocab['OPEN']
            for w in words:
                O_w = O_w * self.vocab[self.total_map[w]]
            O += O_w.v
        if re.match(".*pick up the .*", mission):
            words = mission.partition("pick up the ")[2].split(" ")
            O_w = self.vocab['PICK_UP']
            for w in words:
                O_w = O_w * self.vocab[self.total_map[w]]
            O += O_w.v
        if re.match(".*pick up a .*", mission):
            words = mission.partition("pick up the ")[2].split(" ")
            O_w = self.vocab['PICK_UP']
            for w in words:
                O_w = O_w * self.vocab[self.total_map[w]]
            O += O_w.v
        if re.match(".*put the .* next to the .*", mission):
            pre_words = mission.partition("put the ")[2]
            words = pre_words.partition(" next to the ")[0].split(" ")
            O_w = self.vocab['PUT']
            for w in words:
                O_w = O_w * self.vocab[self.total_map[w]]
            O += O_w.v
            words = pre_words.partition(" next to the ")[2].split(" ")
            O_w = self.vocab['NEXT_TO']
            for w in words:
                O_w = O_w * self.vocab[self.total_map[w]]
            O += O_w.v
            
        O = O / np.linalg.norm(O) 
        return {
            'mission': obs['mission'],
            'image': np.hstack([M, O]).reshape(-1)
        }


