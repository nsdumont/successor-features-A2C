import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
#import torch_rbf as rbf
import numpy as np
#from utils.ssps import *
from gym.spaces import Discrete, Box

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
        #m.weight.data.fill_(0)
        #m.weight.data.normal_(0, 1)
        #m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
           # m.bias.data.fill_(0)
            m.bias.data.normal_(0,1)
            m.bias.data *= 1 / torch.sqrt(m.bias.data.pow(2).sum())

    

class SRModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False,
                 input_type="image", feature_learn="curiosity"):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.n_actions = action_space.n
        self.feature_learn = feature_learn

        if input_type == "image":
            self.feature_in = ImageInput(obs_space,use_memory=use_memory,use_text=use_text)
            self.goal_embedding_size = self.feature_in.other.text_embedding_size
        elif input_type=="flat":
            self.feature_in = FlatInput(obs_space,use_memory=use_memory,use_text=use_text)
            self.goal_embedding_size = self.feature_in.other.text_embedding_size
        elif input_type=="ssp":
            self.feature_in = InputModule(obs_space,obs_space["image"][0],use_memory=use_memory,use_text=use_text)
            self.goal_embedding_size = self.feature_in.input_embedding_size

            
        self.image_embedding_size = self.feature_in.input_embedding_size
        self.embedding_size = self.feature_in.embedding_size
            
        if feature_learn=="reconstruction" and input_type=="image":
            self.feature_out = ImageReconstruction()
        elif feature_learn=="reconstruction" and input_type=="flat":
            self.feature_out = FlatReconstruction(obs_space["image"].shape[0])
        elif feature_learn=="curiosity":
            self.feature_out = Curiosity(self.embedding_size,self.n_actions)
        
        
        # Define reward model
        #self.reward2 = nn.Linear(self.embedding_size, 1, bias=False)
        #self.reward = nn.ModuleList([nn.Linear(self.goal_embedding_size, self.embedding_size)])
        #self.reward = nn.ModuleList([nn.Linear(self.image_embedding_size, self.embedding_size, bias=True)])
        #self.reward = nn.Module()
        #self.w = torch.nn.Parameter(torch.randn(1,self.embedding_size))
        #self.w.requires_grad = True
        #self.reward.register_parameter(name='w', param=nn.Parameter(torch.randn(1,self.embedding_size), requires_grad = True))

        # Define SR model
        self.SR = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh()
        )
        
        
        # Define actor's model
        if type(action_space) == Box:
            self.actor = ContinuousActor(self.embedding_size,self.n_actions)
            self.continuous_action = True
        else:
            self.actor = DiscreteActor(self.embedding_size,self.n_actions)
            self.continuous_action = False
        

        # Initialize parameters correctly
        self.apply(init_params)
        self.reward = nn.Linear(self.goal_embedding_size, self.embedding_size)
        self.reward.apply(init_params2)
 

    def forward(self, obs, action=None, next_obs=None, memory=None):
        embedding, memory, embed_goal = self.feature_in(obs, memory=memory)
        if action is not None:
             next_embedding, _, _ = self.feature_in(next_obs, memory=memory)
             predictions = self.feature_out(embedding, next_embedding = next_embedding,action=action, next_obs=next_obs, memory=memory)

        else:
            predictions = None
        
        dist = self.actor(embedding)
        
        successor = self.SR(embedding) + embedding
        
        #w=0
        #w = self.reward.w #
        #w = self.reward[0](embed_goal)
        r_vec = self.reward(embed_goal)
        reward =  torch.sum(r_vec * embedding,1)
        #reward2 = self.reward2(embedding).squeeze(1)
        
        with torch.no_grad():
            #value = self.reward(successor) 
            #value = value.squeeze(1)
            value = (r_vec * successor).sum(-1)

        return dist, value, embedding, predictions, successor, reward, r_vec, memory


    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size 



### Modules for getting feature representation from raw input

## Add-ons that use memory (LSTMs) and text input
class InputModule(nn.Module):
    def __init__(self,obs_space, input_embedding_size, use_memory, use_text):
        super(InputModule, self).__init__()
        self.use_text = use_text
        self.use_memory = use_memory
        self.input_embedding_size = input_embedding_size
        
        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.input_embedding_size, self.input_embedding_size)

        # Define text embedding
        if self.use_text:
            self.text_embedding_size = input_embedding_size

           # self.word_embedding_size = 2#32
            #self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            
            self.text_layer = nn.Linear(obs_space["text"][0], self.text_embedding_size)
            #self.text_layer = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)
        else:
            self.text_embedding_size = 1

        # Resize image embedding
        self.embedding_size = self.input_embedding_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size
    
    def _get_embed_text(self, text):
        #_, hidden = self.text_layer(self.word_embedding(text))
        #return hidden[-1]
        return F.relu(self.text_layer(text))
        
    

    @property
    def semi_memory_size(self):
        return self.input_embedding_size
        
    def forward(self, obs, x=None, memory=None):
        if x is None:
            x = obs.image
        
        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size].clone(), memory[:, self.semi_memory_size:].clone())
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0].clone()
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)
        else:
            embed_text = torch.zeros((obs.text.shape[0],self.text_embedding_size))
            
        return embedding, memory, embed_text
    
## Features from image data 
class ImageInput(nn.Module):
    def __init__(self, obs_space, use_memory, use_text):
        super(ImageInput, self).__init__()
        self.use_text = use_text
        self.use_memory = use_memory
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.input_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.Tanh()
        )
        
        self.other = InputModule(obs_space, self.input_embedding_size, use_memory=use_memory, use_text=use_text )
        self.embedding_size = self.other.embedding_size
        
    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        embedding = self.image_conv(x)
        x = embedding.reshape(embedding.shape[0], -1)
        
        embedding, memory, embed_goal = self.other(obs, x, memory)
            
        return embedding, memory, embed_goal
    
   
## Features from flat input (e.g. one hot, ssps)
class FlatInput(nn.Module):
    def __init__(self, obs_space,use_memory,  use_text, input_embedding_size=200, hidden_size=256):
        super(FlatInput, self).__init__()
        self.input_dim = obs_space["image"][0]
        self.input_embedding_size = self.input_dim# input_embedding_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_embedding_size),#nn.Tanh()
            nn.Tanh()
        )
        self.other = InputModule(obs_space, self.input_embedding_size, use_text, use_memory)
        self.embedding_size = self.other.embedding_size
        
    def forward(self, obs, memory):
        x = obs.image
        x = self.layers(x)
        embedding, memory, embed_goal = self.other(obs, x, memory)
            
        return embedding, memory, embed_goal
    

    
### Modules for getting predictions used for feature learning

## Auto encoder type
class ImageReconstruction(nn.Module):
    def __init__(self):
        super(ImageReconstruction, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, (4, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, (2, 2))
        )
        
    def forward(self, embedding, **kwargs):
        obs_pred = self.decoder(embedding.reshape(-1,64,1,1))
        obs_pred = obs_pred.transpose(3, 2).transpose(1, 3)
        return obs_pred
    
    
class FlatReconstruction(nn.Module):
    def __init__(self, output_size):
        super(ImageReconstruction, self).__init__()
        self.decoder = nn.Sequential(
            
        )
        
    def forward(self, embedding, **kwargs):
        obs_pred = self.decoder(embedding)
        return obs_pred
    
## curiosity type
class Curiosity(nn.Module):
    def __init__(self,embedding_size,n_actions):
        super(Curiosity, self).__init__()
        self.embedding_size = embedding_size
        self.n_actions = n_actions
        self.forward_model = nn.Sequential(
            nn.Linear(self.embedding_size + self.n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, self.embedding_size),
            nn.Tanh()
            )
        
        self.inverse_model = nn.Sequential(
            nn.Linear(self.embedding_size*2, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions),
            nn.LogSigmoid()
            )
        
    def forward(self, embedding, next_embedding, action, next_obs, memory):
        if self.n_actions > 1:
            action = F.one_hot(action.long(), num_classes=self.n_actions).float()
        else:
            action = action.float()
        forward_input = torch.cat((embedding, action), 1)
        next_obs_pred = self.forward_model(forward_input)
        
        inverse_input = torch.cat((embedding, next_embedding), 1)
        action_pred = self.inverse_model(inverse_input)

        return [next_embedding, next_obs_pred, action_pred]
    

## Actor Modules
class DiscreteActor(nn.Module):
    def __init__(self,embedding_size, n_actions):
        super(DiscreteActor, self).__init__()
        self.n_actions = n_actions
        self.embedding_size = embedding_size
        self.actor_layers = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.n_actions)
        )
        
        
    def forward(self, embedding):
        x = self.actor_layers(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        return dist
    
class ContinuousActor(nn.Module):
    def __init__(self,embedding_size, n_actions):
        super(ContinuousActor, self).__init__()
        self.embedding_size = embedding_size
        self.n_actions = n_actions
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 40),
            nn.ReLU()
        )
        
        self.mean = nn.Sequential(
            nn.Linear(40, self.n_actions),
        )
        
        self.var = nn.Sequential(
            nn.Linear(40, self.n_actions),
            nn.Softplus()
        )
        
    def forward(self, embedding):
        x = self.actor(embedding)
        mean = self.mean(x)
        scale = self.var(x) + 1e-7
        dist = torch.distributions.normal.Normal(mean, scale)
        return dist