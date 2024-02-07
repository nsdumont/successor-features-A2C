import torch
import torch.nn as nn
import torch_ac
#import torch_rbf as rbf
import numpy as np
#from utils.ssps import *
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

from .submodels import ImageInput, FlatInput, SSPInput,  IdentityInput, ContinuousActor, DiscreteActor, SPActor
from .submodels import ImageReconstruction, FlatReconstruction, Curiosity, IdentityOutput
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
            #m.bias.data.normal_(0,1)

def init_params2(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.fill_(0)
        #m.weight.data.normal_(0, 1)
        #m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
           # m.bias.data.fill_(0)
            m.bias.data.normal_(0,1)
            m.bias.data *= 1 / torch.sqrt(m.bias.data.pow(2).sum())

class SRModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False,
                 input_type="image", feature_learn="curiosity",obs_space_sampler=None):
        super().__init__()
        
        
        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        
        self.reward_clip_min = 1e-5
        self.reward_clip_max = 1e5
        
        # Form of input to model and the method for learning features
        self.input_type = input_type
        self.feature_learn = feature_learn

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
        self.goal_embedding_size = self.feature_in.other.text_embedding_size
        
        # Define actor's model
        if type(action_space) == Box:
            import sklearn.preprocessing
            self.n_actions = action_space.shape[0]
            self.actor = ContinuousActor(self.embedding_size,self.n_actions)
            self.continuous_action = True
            state_space_samples = np.array([obs_space_sampler.sample() for x in range(1000)])
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(state_space_samples)
            self.scaler = scaler
        elif type(action_space) == Discrete:
            self.n_actions = action_space.n
            self.actor = DiscreteActor(self.embedding_size,self.n_actions)
            self.continuous_action = False
        elif type(action_space) == SSPDiscrete:
            self.n_actions = action_space.shape_out
            self.actor = SPActor(self.embedding_size,self.n_actions,action_space)
            self.continuous_action = False

            
        if feature_learn=="reconstruction" and input_type=="image":
            self.feature_out = ImageReconstruction()
        elif feature_learn=="reconstruction" and input_type=="flat":
            self.feature_out = FlatReconstruction(obs_space["image"].shape[0])
        elif feature_learn=="curiosity":
            self.feature_out = Curiosity(self.embedding_size,self.n_actions)
        else:
            self.feature_out = IdentityOutput()
        
        
        # Define SR model
        if input_type.startswith('ssp'):
            self.SR = nn.Sequential(
                nn.Linear(self.embedding_size, 2*self.embedding_size),
                nn.Tanh(),
                nn.Linear(2*self.embedding_size, self.embedding_size),
                nn.Tanh()
            )
        else:
            self.SR = nn.Sequential(
            nn.Linear(self.embedding_size, 2*self.embedding_size),
            nn.Tanh(),
            nn.Linear(2*self.embedding_size, self.embedding_size)
        )
        
        #nn.Linear(self.embedding_size, self.embedding_size)
        

        # Initialize parameters correctly
        self.reward = nn.Linear(self.embedding_size, 1, bias=False)
        self.apply(init_params)
        
        # 
        # self.reward = nn.Linear(self.goal_embedding_size, self.embedding_size)
        # self.reward = torch.nn.Parameter(torch.zeros(self.embedding_size,1))
        # self.reward.apply(init_params2)
        


 

    def forward(self, obs, action=None, next_obs=None, memory=None):
        embedding, memory, embed_txt = self.feature_in(obs, memory=memory)
        if (action is not None) and (next_obs is not None):
             next_embedding, _, _ = self.feature_in(next_obs, memory=memory)
             predictions = self.feature_out(embedding, next_embedding = next_embedding, action=action, next_obs=next_obs, memory=memory)
        else:
            predictions = None
        
        dist = self.actor(embedding)
        successor = self.SR(embedding) + embedding # skip connection
        # reward = self.reward(embedding).squeeze() 
        # reward_vector = self.reward(embed_txt)
        reward_vector = self.reward.weight/ torch.clamp(torch.norm(self.reward.weight), self.reward_clip_min,self.reward_clip_max)
        reward = torch.sum(reward_vector * embedding,1)
        
        # value = self.reward(successor).squeeze().detach()
        value = torch.sum(reward_vector * successor,1).detach()

        return dist, value, embedding, predictions, successor, reward, memory


    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size 


