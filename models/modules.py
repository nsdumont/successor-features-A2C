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
import ast

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
    def __init__(self, obs_space, input_embedding_size, use_memory, use_text, normalize, **kwargs):
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
        if (n==7) & (m==7): # minigrid
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
        elif (n==60) & (m==80): # miniworld
            self.conv_output_size = 32 * 7 * 5
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=4, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            raise ValueError(f"CNN not set-up for image input of size {n}x{m}. Please add a new option in ImageProcesser")

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
                 input_embedding_size=151, hidden_size=0, activation_fun='relu', basis_type='hex',rng=np.random.default_rng(0), **kwargs):
        super().__init__()
        if type(obs_space["image"]) is int:
            self.input_dim = obs_space["image"]
        else:
            self.input_dim = obs_space["image"][0]
            
        if 'ssp_dim' in kwargs:
            input_embedding_size = kwargs['ssp_dim']
        else:
            input_embedding_size = input_embedding_size
  
          
        if 'ssp_h' in kwargs:
            initial_ls = kwargs['ssp_h']
            if (type(initial_ls) is np.ndarray):
                initial_ls = torch.Tensor(initial_ls.flatten())
            elif (type(initial_ls) is list):
                initial_ls = torch.Tensor(initial_ls)
            elif type(initial_ls) is str:
                initial_ls = torch.Tensor(ast.literal_eval(initial_ls))
        else:
            initial_ls = 1.0
        
       
        # if basis_type=='hex':
        #     ssp_space = HexagonalSSPSpace(self.input_dim, input_embedding_size)
        #     self.phase_matrix = torch.nn.Parameter(torch.Tensor(ssp_space.phase_matrix),requires_grad=False)
        #     self.length_scale = torch.nn.Parameter(initial_ls*torch.ones(self.input_dim), requires_grad=True)
        # elif basis_type=='rand':
        #     ssp_space = RandomSSPSpace(self.input_dim, input_embedding_size)
        #     self.phase_matrix = torch.nn.Parameter(torch.Tensor(ssp_space.phase_matrix),requires_grad=False)
        #     self.length_scale = torch.nn.Parameter(initial_ls*torch.ones(self.input_dim), requires_grad=True)
        # elif basis_type=='learn':
        #     ssp_space = HexagonalSSPSpace(self.input_dim, input_embedding_size) # initial
        #     self.phase_matrix = torch.nn.Parameter(torch.Tensor(ssp_space.phase_matrix),requires_grad=True)
        #     self.length_scale = torch.nn.Parameter(initial_ls*torch.ones(self.input_dim), requires_grad=True)
            # raise Exception("Learning the full phase matrix of the SSP encosing is not yet implemented") 
        
        if basis_type=='hex':
            ssp_space = HexagonalSSPSpace(self.input_dim, input_embedding_size,rng=rng)
        elif basis_type=='rand':
            ssp_space = RandomSSPSpace(self.input_dim, input_embedding_size, rng=rng)
            
        self._features_dim = ssp_space.ssp_dim
        self.nparams = (self._features_dim-1)//2
        self.phase_matrix = nn.Parameter(torch.Tensor(ssp_space.phase_matrix[1:(self.nparams+1),:]),requires_grad=False)
        self.length_scale = nn.Parameter(initial_ls*torch.ones(self.input_dim), requires_grad=True)
        
        if hidden_size>0:
            self.layers =  mlp(ssp_space.ssp_dim, hidden_size, activation_fun, input_embedding_size)
            self.input_embedding_size = input_embedding_size
        else:
            self.layers = torch.nn.Identity()
            self.input_embedding_size = ssp_space.ssp_dim

        
        self.other = FeatureProcesser(obs_space, self.input_embedding_size, 
                                 use_text=use_text, use_memory=use_memory, normalize=normalize)
        self.embedding_size = self.other.embedding_size
        
    def _encode(self, x):
        ls_mat = torch.atleast_2d(torch.diag(1/self.length_scale)).to(x.device)
        F = torch.zeros((self._features_dim, self.input_dim)).to(x.device)
        F[1:(self.nparams+1),:] = self.phase_matrix
        F[(self.nparams+1):,:] = -torch.flip(self.phase_matrix, dims=(0,))
        x = (F @ (x @ ls_mat).T).type(torch.complex64) # fix .to(x.device)
        x = torch.fft.ifft( torch.exp( 1.j * x), axis=0 ).real.T
        return x
        
    def forward(self, obs, memory):
        x = self._encode(obs.image)
        x = self.layers(x)
        embedding, memory, embed_txt = self.other(obs, x, memory)
        return embedding, memory, embed_txt
    
class SSPViewProcesser(nn.Module): # only for minigrid
    def __init__(self, obs_space,use_memory,  use_text, normalize, 
                 input_embedding_size=151, hidden_size=0, activation_fun='relu', basis_type='hex', rng=np.random.default_rng(0), **kwargs):
        super().__init__()
        self.input_dim = 3
            
        if 'ssp_dim' in kwargs:
            input_embedding_size = kwargs['ssp_dim']
        else:
            input_embedding_size = input_embedding_size
  
          
        if 'ssp_h' in kwargs:
            initial_ls = kwargs['ssp_h']
            if (type(initial_ls) is np.ndarray):
                initial_ls = torch.Tensor(initial_ls.flatten())
            elif (type(initial_ls) is list):
                initial_ls = torch.Tensor(initial_ls)
            elif type(initial_ls) is str:
                initial_ls = torch.Tensor(ast.literal_eval(initial_ls))
        else:
            initial_ls = 1.0
        
       
        if basis_type=='hex':
            ssp_space = HexagonalSSPSpace(self.input_dim, input_embedding_size,rng=rng)
        elif basis_type=='rand':
            ssp_space = RandomSSPSpace(self.input_dim, input_embedding_size, rng=rng)
            
        self._features_dim = ssp_space.ssp_dim
        self.nparams = (self._features_dim-1)//2
        self.phase_matrix = nn.Parameter(torch.Tensor(ssp_space.phase_matrix[1:(self.nparams+1),:]),requires_grad=False)
        self.length_scale = nn.Parameter(initial_ls*torch.ones(self.input_dim), requires_grad=True)
        
        self.view_width = 7
        self.view_height = 7
        domain_bounds = np.array([ [0, self.view_width-1],
                                  [-(self.view_height-1)//2, (self.view_height-1)//2 ],
                                  [0,3]])
        xs = [np.arange(domain_bounds[i,1],domain_bounds[i,0]-1,-1) for i in range(2)]
        xx = np.meshgrid(*xs)
        xx[0] = 3 - xx[0]
        xx[1] = 6 - xx[0]
        self.grid_pts = torch.Tensor(np.array(xx))
        self.n_pts = self.grid_pts.shape[1] * self.grid_pts.shape[2]
        self.unroll_grid_pts  = torch.vstack([self.grid_pts[0,:].flatten(), self.grid_pts[1,:].flatten(), -torch.ones(self.n_pts)]).T
        
        if hidden_size>0:
            self.layers =  mlp(ssp_space.ssp_dim, hidden_size, activation_fun, input_embedding_size)
            self.input_embedding_size = input_embedding_size
        else:
            self.layers = torch.nn.Identity()
            self.input_embedding_size = ssp_space.ssp_dim

        
        self.other = FeatureProcesser(obs_space, self.input_embedding_size, 
                                 use_text=use_text, use_memory=use_memory, normalize=normalize)
        self.embedding_size = self.other.embedding_size
        
   
    def _encode(self, x):
        ls_mat = torch.atleast_2d(torch.diag(1/self.length_scale)).to(x.device)
        F = torch.zeros((self._features_dim, self.input_dim)).to(x.device)
        F[1:(self.nparams+1),:] = self.phase_matrix
        F[(self.nparams+1):,:] = -torch.flip(self.phase_matrix, dims=(0,))
        x = (F @ (x @ ls_mat).T).type(torch.complex64) # fix .to(x.device)
        x = torch.fft.ifft( torch.exp( 1.j * x), axis=0 ).real.T
        return x
    
    def _bind(self,a,b):
        return torch.fft.ifft(torch.fft.fft(a, axis=-1) * torch.fft.fft(b, axis=-1), axis=-1).real
        
    def forward(self, obs, memory):
        x = obs.image[:,:self.input_dim]
        M = self._encode(x)
        
        obj_sps = obs.image[:,self.input_dim:-self._features_dim].reshape(obs.image.shape[0],-1,self._features_dim)
        ssp_grid_pts = self._encode(self.unroll_grid_pts.to(x.device)).reshape(1, self.n_pts, self._features_dim)
        M += torch.sum(self._bind(obj_sps, ssp_grid_pts), axis=1)     
        M += obs.image[:,-self._features_dim:]
        
        x = self.layers(M)
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
        self.actor_layers = mlp(embedding_size, hidden_size, "relu",
                                hidden_size, "relu")
        self.mu = nn.Linear(hidden_size, n_actions)
        self.log_sigma = nn.Linear(hidden_size, n_actions)
        
        
    def forward(self, embedding):
        x= self.actor_layers(embedding)
        mu= self.mu(x)
        assert not torch.isnan(x).any()
        assert not torch.isnan(mu).any()
        log_std = self.log_sigma(x)
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