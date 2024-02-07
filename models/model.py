import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box
from .submodels import ImageInput, FlatInput, SSPInput, IdentityInput, ContinuousActor, DiscreteActor, SPActor
import sys,os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from spaces import SSPBox, SSPDiscrete

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False,input_type="image",obs_space_sampler=None):
        super().__init__()
        
        
        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        
        # Form of input to model and the method for learning features
        self.input_type = input_type

        if input_type == "image":
            self.feature_in = ImageInput(obs_space,use_memory=use_memory,use_text=use_text)
        elif input_type=="flat":
            self.feature_in = FlatInput(obs_space,use_memory=use_memory,use_text=use_text)
        elif input_type=="ssp":
            self.feature_in = SSPInput(obs_space,use_memory=use_memory,use_text=use_text)
        elif input_type=="none":
            self.feature_in = IdentityInput(obs_space,use_memory=use_memory,use_text=use_text)
        else:
            raise ValueError("Incorrect input type name: {}".format(input_type))
            
        self.image_embedding_size = self.feature_in.input_embedding_size
        self.embedding_size = self.feature_in.embedding_size
        
        # Define actor's model
        if type(action_space) == Box:
            import sklearn.preprocessing
            self.n_actions = action_space.shape[0]
            self.actor = ContinuousActor(self.embedding_size,self.n_actions)
            self.continuous_action = True
            state_space_samples = np.array([obs_space_sampler.sample() for x in range(10000)])
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(state_space_samples)
            self.scaler = scaler
        elif type(action_space) == Discrete:
            self.n_actions = action_space.n
            self.actor = DiscreteActor(self.embedding_size,self.n_actions)
            self.continuous_action = False
        elif type(action_space) == SSPDiscrete:
            self.n_actions = action_space.shape_out
            self.actor = SPActor(self.embedding_size,self.n_actions, action_space)
            self.continuous_action = False

        
        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        embedding, memory,_ = self.feature_in(obs, memory=memory)

        dist = self.actor(embedding)

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
