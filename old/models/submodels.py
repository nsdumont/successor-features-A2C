import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np


### Modules for getting feature representation from raw input

## Add-ons that use memory (LSTMs) and text input
class InputModule(nn.Module):
    def __init__(self, obs_space, input_embedding_size, use_memory, use_text):
        super(InputModule, self).__init__()
        self.use_text = use_text
        self.use_memory = use_memory
        self.input_embedding_size = input_embedding_size
        
        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.input_embedding_size, self.input_embedding_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.input_embedding_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size
        
    
    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
        
    @property
    def semi_memory_size(self):
        return self.input_embedding_size
        
    def forward(self, obs, x, memory=None):
        if self.use_memory & (memory is not None):
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)
        
        return embedding, memory
    
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
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        embedding, memory = self.other(obs, x, memory)
        return embedding, memory
    
   
## Features from flat input 
class FlatInput(nn.Module):
    def __init__(self, obs_space,use_memory,  use_text, input_embedding_size=10, hidden_size=256):
        super(FlatInput, self).__init__()
        if type(obs_space["image"]) is int:
            self.input_dim = obs_space["image"]
        else:
            self.input_dim = obs_space["image"][0]
            
        self.input_embedding_size = input_embedding_size
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.input_embedding_size),
        )
        self.other = InputModule(obs_space, self.input_embedding_size, 
                                 use_text=use_text, use_memory=use_memory)
        self.embedding_size = self.other.embedding_size
        
    def forward(self, obs, memory):
        x = obs.image
        x = self.layers(x)
        embedding, memory = self.other(obs, x, memory)
        return embedding, memory
    
## Features from flat input 
class IdentityInput(nn.Module):
    def __init__(self, obs_space, use_memory,  use_text):
        super(IdentityInput, self).__init__()
        self.input_embedding_size =  obs_space["image"]
        self.other = InputModule(obs_space, self.input_embedding_size, 
                                 use_text=use_text, use_memory=use_memory)
        self.embedding_size = self.other.embedding_size
        
    def forward(self, obs, memory):
        x = obs.image
        embedding, memory = self.other(obs, x, memory)
        return embedding, memory
 
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
            nn.Linear(output_size, output_size),
            nn.Tanh(),
            nn.Linear(output_size, output_size)
        )
        
    def forward(self, embedding, **kwargs):
        obs_pred = self.decoder(embedding)
        return obs_pred
    
class IdentityOutput(nn.Module):
    def __init__(self):
        super(IdentityOutput, self).__init__()
        
    def forward(self, embedding, **kwargs):
        return embedding
    
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
            action = action.float().reshape(-1,1)
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
    def __init__(self,embedding_size, n_actions, hidden_size=40):
        super(ContinuousActor, self).__init__()
        self.embedding_size = embedding_size
        self.n_actions = n_actions
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, hidden_size),
            nn.ReLU()
        )
        
        self.mean = nn.Sequential(
            nn.Linear(hidden_size, self.n_actions),
            nn.Tanh(),
        )
        
        self.var = nn.Sequential(
            nn.Linear(hidden_size, self.n_actions),
            nn.Sigmoid(),
            
        )
        
    def forward(self, embedding):
        x = self.actor(embedding)
        mean = self.mean(x)
        scale = self.var(x) + 1e-16
        dist = torch.distributions.normal.Normal(mean, scale)
        return dist