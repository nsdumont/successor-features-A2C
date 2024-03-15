import numpy as np
import typing as tp
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import sys,os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from spaces import HexagonalSSPSpace, RandomSSPSpace
from distributions import SPDistribution
from utils import weight_init, SquashedNormal

class _L2(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = torch.sqrt(self.dim) * F.normalize(x, dim=1)
        return y

def _nl(name: str, dim: int) -> tp.List[nn.Module]:
    """Returns a non-linearity given name and dimension"""
    if name == "irelu":
        return [nn.ReLU(inplace=True)]
    if name == "relu":
        return [nn.ReLU()]
    if name == "ntanh":
        return [nn.LayerNorm(dim), nn.Tanh()]
    if name == "layernorm":
        return [nn.LayerNorm(dim)]
    if name == "tanh":
        return [nn.Tanh()]
    if name == "L2":
        return [_L2(dim)]
    raise ValueError(f"Unknown non-linearity {name}")


def mlp(*layers: tp.Sequence[tp.Union[int, str]]) -> nn.Sequential:
    """Provides a sequence of linear layers and non-linearities
    providing a sequence of dimension for the neurons, or name of
    the non-linearities
    Eg: mlp(10, 12, "relu", 15) returns:
    Sequential(Linear(10, 12), ReLU(), Linear(12, 15))
    """
    assert len(layers) >= 2
    sequence: tp.List[nn.Module] = []
    assert np.issubdtype(type(layers[0]), int), "First input must provide the dimension"
    prev_dim: int = layers[0]
    for layer in layers[1:]:
        if isinstance(layer, str):
            sequence.extend(_nl(layer, prev_dim))
        else:
            assert np.issubdtype(type(layer), int)
            sequence.append(nn.Linear(prev_dim, layer))
            prev_dim = layer
    return nn.Sequential(*sequence)

    
    

### Modules for getting feature representation from raw input


## Add-ons that use memory (LSTMs) and text input
class FeatureProcesser(nn.Module):
    def __init__(self, obs_space, input_embedding_size, use_memory, use_text, normalize):
        super().__init__()
        self.use_text = use_text
        self.use_memory = use_memory
        self.normalize = normalize
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
        else:
            self.text_embedding_size = 128
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
        else:
            embed_text = torch.zeros((len(obs),self.text_embedding_size)).to(embedding.device)
        
        if self.normalize:
            embedding = F.normalize(x, p=2, dim=1) 
        return embedding, memory, embed_text
    
## Features from image data 
class ImageProcesser(nn.Module):
    def __init__(self, obs_space, use_memory, use_text,normalize,  input_embedding_size=64, **kwargs):
        super().__init__()
        self.use_text = use_text
        self.use_memory = use_memory
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        
        self.input_embedding_size = input_embedding_size
        self.conv_output_size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.linear = nn.Sequential(nn.Linear(self.conv_output_size, self.input_embedding_size), nn.ReLU())
        self.other = FeatureProcesser(obs_space, self.input_embedding_size,
                                      use_memory=use_memory, use_text=use_text, normalize=normalize )
        self.embedding_size = self.other.embedding_size
        
    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.linear(self.cnn(x))
        embedding, memory, embed_txt = self.other(obs, x, memory)
        return embedding, memory, embed_txt
    
   
## Features from flat input 
class FlatProcesser(nn.Module):
    def __init__(self, obs_space, use_memory,  use_text, normalize, 
                 input_embedding_size=64, hidden_size=256, activation_fun='tanh', **kwargs):
        super().__init__()
        if type(obs_space["image"]) is int:
            self.input_dim = obs_space["image"]
        else:
            self.input_dim = obs_space["image"][0]
            
        self.input_embedding_size = input_embedding_size
        self.hidden_size = hidden_size
        # self.layers = mlp(self.input_dim, self.hidden_size, "ntanh", self.input_embedding_size, "irelu")
        self.layers = mlp(self.input_dim, self.hidden_size, activation_fun, self.input_embedding_size) #tanh
        self.other = FeatureProcesser(obs_space, self.input_embedding_size, 
                                 use_text=use_text, use_memory=use_memory, normalize=normalize)
        self.embedding_size = self.other.embedding_size
        
    def forward(self, obs, memory):
        x = obs.image
        x = self.layers(x)
        embedding, memory, embed_txt = self.other(obs, x, memory)
        return embedding, memory, embed_txt
   
## Construct SSP features from input with trainable ls  
class SSPProcesser(nn.Module):
    def __init__(self, obs_space,use_memory,  use_text, normalize, 
                 input_embedding_size=151, hidden_size=0, activation_fun='relu', basis_type='hex', **kwargs):
        super().__init__()
        if type(obs_space["image"]) is int:
            self.input_dim = obs_space["image"]
        else:
            self.input_dim = obs_space["image"][0]
            
        if basis_type=='hex':
            ssp_space = HexagonalSSPSpace(self.input_dim, input_embedding_size)
            self.phase_matrix = torch.nn.Parameter(torch.Tensor(ssp_space.phase_matrix),requires_grad=False)
            self.length_scale = torch.nn.Parameter(torch.ones(self.input_dim), requires_grad=True)
        elif basis_type=='rand':
            ssp_space = RandomSSPSpace(self.input_dim, input_embedding_size)
            self.phase_matrix = torch.nn.Parameter(torch.Tensor(ssp_space.phase_matrix),requires_grad=False)
            self.length_scale = torch.nn.Parameter(torch.ones(self.input_dim), requires_grad=True)
        elif basis_type=='learn':
            ssp_space = HexagonalSSPSpace(self.input_dim, input_embedding_size) # initial
            self.phase_matrix = torch.nn.Parameter(torch.Tensor(ssp_space.phase_matrix),requires_grad=True)
            self.length_scale = torch.nn.Parameter(torch.ones(self.input_dim), requires_grad=False)
            # raise Exception("Learning the full phase matrix of the SSP encosing is not yet implemented") 
        
        
        if hidden_size>0:
            self.layers =  mlp(ssp_space.ssp_dim, hidden_size, activation_fun, input_embedding_size)
            self.input_embedding_size = input_embedding_size
        else:
            self.layers = torch.nn.Identity()
            self.input_embedding_size = ssp_space.ssp_dim

        
        self.other = FeatureProcesser(obs_space, self.input_embedding_size, 
                                 use_text=use_text, use_memory=use_memory, normalize=normalize)
        self.embedding_size = self.other.embedding_size
        
    def forward(self, obs, memory):
        x = obs.image
        ls_mat = torch.atleast_2d(torch.diag(self.length_scale))
        x = (self.phase_matrix @ (x @ ls_mat).T).type(torch.complex64) # fix .to(x.device)
        x = torch.fft.ifft( torch.exp( 1.j * x), axis=0 ).real.T
        x = self.layers(x)
        embedding, memory, embed_txt = self.other(obs, x, memory)
        return embedding, memory, embed_txt
    
## Features are the input, no transform
class IdentityProcesser(nn.Module):
    def __init__(self, obs_space, use_memory, use_text, normalize, **kwargs):
        super().__init__()
        self.input_embedding_size =  obs_space["image"]
        self.other = FeatureProcesser(obs_space, self.input_embedding_size, 
                                 use_text=use_text, use_memory=use_memory,normalize=normalize)
        self.embedding_size = self.other.embedding_size
        
    def forward(self, obs, memory):
        x = obs.image
        embedding, memory, embed_txt = self.other(obs, x, memory)
        return embedding, memory, embed_txt
 


## Actor Modules
class DiscreteActor(nn.Module):
    def __init__(self,embedding_size, n_actions, hidden_size=64):
        super(DiscreteActor, self).__init__()
        self.n_actions = n_actions
        self.embedding_size = embedding_size
        self.actor_layers = mlp(embedding_size, hidden_size, "irelu", n_actions)
        
    def forward(self, embedding):
        x = self.actor_layers(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))
        return dist
    
class ContinuousActor(nn.Module):
    def __init__(self,embedding_size, n_actions, hidden_size=64,log_std_bounds=(-5,2)):
        super(ContinuousActor, self).__init__()
        self.embedding_size = embedding_size
        self.n_actions = n_actions
        self.log_std_bounds=log_std_bounds
        self.actor_layers = mlp(embedding_size, hidden_size, "ntanh",
                                hidden_size, "relu", 2 * n_actions)
        
        
    def forward(self, embedding):
        mu, log_std = self.actor_layers(embedding).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        dist = SquashedNormal(mu, std)
        return dist
    
# class SPActor(nn.Module):
#     def __init__(self,embedding_size, n_actions, action_space, hidden_size=64):
#         super(SPActor, self).__init__()
#         self.embedding_size = embedding_size
#         self.action_space=action_space
#         self.n_actions = n_actions
#         self.actor_layers = nn.Sequential(
#             nn.Linear(self.embedding_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, self.n_actions),
#             nn.Tanh(),
#         )
          
#     def forward(self, embedding):
#         x = self.actor_layers(embedding)
#         dist = SPDistribution(x,self.action_space)
#         return dist