import torch.nn as nn
import torch_ac
import numpy as np
from gymnasium.spaces import Discrete, Box
from .modules import mlp, ImageProcesser, FlatProcesser, SSPProcesser, SSPViewProcesser, IdentityProcesser
from .modules import ContinuousActor, DiscreteActor
import sys,os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from spaces import SSPBox, SSPDiscrete
from utils import weight_init

feature_rep_options = {'image': ImageProcesser, 'flat': FlatProcesser, 'none': IdentityProcesser,
                       'ssp': SSPProcesser, 'ssp-view': SSPViewProcesser}

class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False,
                 use_text=False,normalize=False,input_type="image",obs_space_sampler=None,
                 critic_hidden_size=64, actor_hidden_size=64, 
                 feature_hidden_size=256, feature_size=64, **kwargs):
        super().__init__()
        
        
        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        
        # Form of input to model and the method for learning features
        self.input_type = input_type

        if input_type in feature_rep_options.keys():
            self.feature_net = feature_rep_options[input_type](obs_space,use_memory=use_memory,use_text=use_text, normalize=normalize,
                                                              hidden_size=feature_hidden_size,input_embedding_size=feature_size, **kwargs)
        else:
            raise ValueError("Incorrect input type option: {} (should be {})".format(input_type, ', '.join(feature_rep_options.keys())))
            
        self.image_embedding_size = self.feature_net.input_embedding_size
        self.embedding_size = self.feature_net.embedding_size

        
        # Define actor's model
        if type(action_space) == Box:
            self.n_actions = action_space.shape[0]
            self.actor = ContinuousActor(self.embedding_size,self.n_actions,hidden_size=actor_hidden_size)
            self.continuous_action = True
        elif type(action_space) == Discrete:
            self.n_actions = action_space.n
            self.actor = DiscreteActor(self.embedding_size,self.n_actions,hidden_size=actor_hidden_size)
            self.continuous_action = False
        else:
            raise ValueError("Unsupported action space")
           
        # elif type(action_space) == SSPDiscrete:
        #     self.n_actions = action_space.shape_out
        #     self.actor = SPActor(self.embedding_size,self.n_actions, action_space)
        #     self.continuous_action = False

        
        # Define critic's model
        self.critic = mlp(self.embedding_size, critic_hidden_size,  "irelu", 1)
       
        # Initialize parameters 
        self.apply(weight_init)
        

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        embedding, memory,_ = self.feature_net(obs, memory=memory)
        dist = self.actor(embedding)
        value = self.critic(embedding).squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
