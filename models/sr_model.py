import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box
from .modules import mlp, ImageProcesser, FlatProcesser, SSPProcesser, IdentityProcesser, SSPViewProcesser
from .modules import ContinuousActor, DiscreteActor
from .sr_modules import Identity, Laplacian, CM,CMv2
from .sr_modules import ICM,ICMv2, TransitionModel, TransitionLatentModel, AutoEncoder, ImageAutoEncoder
import sys,os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from spaces import SSPBox, SSPDiscrete
from utils import weight_init


feature_rep_options = {'image': ImageProcesser, 'flat': FlatProcesser, 
                       'ssp': SSPProcesser, 'ssp-view': SSPViewProcesser, 'none': IdentityProcesser}

# feature_learn_options = {'none-flat': Identity, 'image-image': Identity,
#                          'lap-flat': Laplacian, 'lap-image': Laplacian,
#                          'icm-flat': ICM, 'icm-image': ICM,
#                          'cm-flat': CM, 'cm-image': CM,
#                          'trans-flat': TransitionModel, #'trans-image'
#                          'latent-flat': TransitionLatentModel, 'latent-image': TransitionLatentModel,
#                          'aenc-flat': AutoEncoder, 'aenc-image': ImageAutoEncoder}
feature_learn_options = {'none': Identity, 'lap': Laplacian,
                         'icm': ICM,'cm': CM, 'trans': TransitionModel, #'trans-image'
                         'latent': TransitionLatentModel, 'aenc': AutoEncoder}

class SRModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False,normalize=True,
                 input_type="image", feature_learn="curiosity",obs_space_sampler=None,
                 critic_hidden_size=64, actor_hidden_size=64, 
                 feature_hidden_size=256, feature_size=64, feature_learn_hidden_size=256, **kwargs):
        super().__init__()
        
        
        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        
        # self.reward_clip_min = 1e-5
        # self.reward_clip_max = 1e5
        
        # Form of input to model and the method for learning features
        self.input_type = input_type
        self.feature_learn = feature_learn

        if input_type in feature_rep_options.keys():
            self.feature_net = feature_rep_options[input_type](obs_space,use_memory=use_memory,use_text=use_text,normalize=normalize,
                                                              hidden_size=feature_hidden_size,input_embedding_size=feature_size, **kwargs)
        else:
            raise ValueError("Incorrect input type option: {} (should be {})".format(input_type, ', '.join(feature_rep_options.keys())))
            
        self.image_embedding_size = self.feature_net.input_embedding_size
        self.goal_embedding_size = self.feature_net.other.text_embedding_size
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
        #     self.actor = SPActor(self.embedding_size,self.n_actions,action_space)
        #     self.continuous_action = False
            
            
        if feature_learn in feature_learn_options.keys():
            if (feature_learn =='aenc') & (input_type=='image'):
                self.feature_learner = ImageAutoEncoder(obs_space["image"],
                                     self.n_actions, self.embedding_size, feature_learn_hidden_size, self.feature_net)
            # if (feature_learn=='icm') & ( type(action_space) == Discrete): # only one that depends on action type
            #     self.feature_learner = ICMv2(obs_space["image"],
            #                              self.n_actions, self.embedding_size, feature_learn_hidden_size, self.feature_net)
            # elif (feature_learn=='cm') & ( type(action_space) == Discrete): # only one that depends on action type
            #     self.feature_learner = CMv2(obs_space["image"],
            #                              self.n_actions, self.embedding_size, feature_learn_hidden_size, self.feature_net)
            
            else:
                self.feature_learner = feature_learn_options[feature_learn](obs_space["image"],
                                     self.n_actions, self.embedding_size, feature_learn_hidden_size, self.feature_net)
        else:
            raise ValueError("Incorrect  feature learn + input type combination: {} (should be one of {})".format(feature_learn+'-'+input_type, ', '.join(feature_learn_options.keys())))


    
        # Define SR model
        self.SR = mlp(self.embedding_size, critic_hidden_size, 'tanh', self.embedding_size)

        
        # Initialize parameters correctly
        self.reward = nn.Linear(self.embedding_size, 1, bias=False)
        self.apply(weight_init)
        self.target_feature_net = copy.deepcopy(self.feature_net)
        self.feature_learner.target_feature_net = self.target_feature_net
        # self.reward = nn.Linear(self.goal_embedding_size, self.embedding_size)
        # self.reward = torch.nn.Parameter(torch.zeros(self.embedding_size,1))
        # self.reward.apply(init_params2)
        
 

    def forward(self, obs: torch.Tensor, action=None, memory=None):
        if (action is not None):
             feature_loss = self.feature_learner(obs[:-1], action[:-1], obs[1:], memory[:-1], memory[1:])
        else:
            feature_loss = None
        
        embedding, memory, embed_txt = self.target_feature_net(obs, memory)
        dist = self.actor(embedding)
        successor = self.SR(embedding) + embedding # skip connection
        reward = self.reward(embedding).squeeze() 
        # reward_vector = self.reward(embed_txt)
        # reward_vector = self.reward.weight/ torch.clamp(torch.norm(self.reward.weight), self.reward_clip_min,self.reward_clip_max)
        # reward = torch.sum(reward_vector * embedding,1)
        
        value = self.reward(successor).squeeze().detach()
        # value = torch.sum(reward_vector * successor,1).detach()

        return dist, value, embedding, feature_loss, successor, reward, memory


    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size 


