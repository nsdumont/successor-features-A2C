import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import mlp
import sys,os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from utils import weight_init


 
### Modules for getting predictions used for feature learning

# Changed to use code from 
#https://github.com/facebookresearch/controllable_agent/blob/main/url_benchmark/agent/sf.py

class FeatureLearner(nn.Module):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, feature_net) -> None:
        super().__init__()
        self.feature_net = feature_net
        self.target_feature_net = None
        #self.apply(weight_init)
        

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, memory=None, next_memory=None):
        return None


class Identity(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, feature_net) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim, feature_net)
        
    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, memory=None, next_memory=None):
        del action
        del next_obs
        return torch.zeros(1)


class Laplacian(FeatureLearner):
    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, memory=None, next_memory=None):
        del action
        phi,_,_ = self.feature_net(obs, memory)
        next_phi,_,_ = self.feature_net(next_obs, next_memory)
        loss = (phi - next_phi).pow(2).mean()
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        return loss



class ICM(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, feature_net) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim, feature_net)

        # self.forward_dynamic_net = mlp(z_dim + action_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', z_dim)
        self.inverse_dynamic_net = mlp(2 * z_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', action_dim, 'tanh')
        #self.apply(weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, memory=None, next_memory=None):
        phi,_,_ = self.feature_net(obs, memory[:-1])
        next_phi,_,_ = self.feature_net(next_obs, memory[1:])
        # predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        # forward_error = (next_phi.detach() - predicted_next_obs).pow(2).mean()
        predicted_action = self.inverse_dynamic_net(torch.cat([phi, next_phi], dim=-1))
        backward_error = (action - predicted_action).pow(2).mean()
        icm_loss = backward_error
        # icm_loss = forward_error + backward_error
        return icm_loss
    
class ICMv2(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, feature_net) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim, feature_net)

        # self.forward_dynamic_net = mlp(z_dim + action_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', z_dim)
        self.inverse_dynamic_net = mlp(2 * z_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', action_dim)
        #self.apply(weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, memory=None, next_memory=None):
        phi,_,_ = self.feature_net(obs, memory[:-1])
        next_phi,_,_ = self.feature_net(next_obs, memory[1:])
        # predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        # forward_error = (next_phi.detach() - predicted_next_obs).pow(2).mean()
        predicted_action = self.inverse_dynamic_net(torch.cat([phi, next_phi], dim=-1))
        predicted_action = F.log_softmax(predicted_action, dim=-1) 
        backward_error =  F.nll_loss(predicted_action, action)
        icm_loss = backward_error
        # icm_loss = forward_error + backward_error
        return icm_loss


class CM(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, feature_net) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim, feature_net)

        self.forward_dynamic_net = mlp(z_dim + action_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', z_dim)
        self.inverse_dynamic_net = mlp(2 * z_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', action_dim, 'tanh')
        #self.apply(weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, memory=None, next_memory=None):
        phi,_,_ = self.feature_net(obs, memory[:-1])
        next_phi,_,_ = self.feature_net(next_obs, memory[1:])
        predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        forward_error = (next_phi.detach() - predicted_next_obs).pow(2).mean()
        predicted_action = self.inverse_dynamic_net(torch.cat([phi, next_phi], dim=-1))
        backward_error = (action - predicted_action).pow(2).mean()
        icm_loss = backward_error
        icm_loss = forward_error + backward_error
        return icm_loss
    
class CMv2(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, feature_net) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim, feature_net)

        self.forward_dynamic_net = mlp(z_dim + action_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', z_dim)
        self.inverse_dynamic_net = mlp(2 * z_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', action_dim)
        #self.apply(weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, memory=None, next_memory=None):
        phi,_,_ = self.feature_net(obs[:-1], memory[:-1])
        next_phi,_,_ = self.feature_net(next_obs, memory[1:])
        predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        forward_error = (next_phi.detach() - predicted_next_obs).pow(2).mean()
        predicted_action = self.inverse_dynamic_net(torch.cat([phi, next_phi], dim=-1))
        predicted_action = F.log_softmax(predicted_action, dim=-1) 
        backward_error =  F.nll_loss(predicted_action, action)
        icm_loss = backward_error
        icm_loss = forward_error + backward_error
        return icm_loss

class TransitionModel(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, feature_net) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim, feature_net)

        self.forward_dynamic_net = mlp(z_dim + action_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', obs_dim)
        #self.apply(weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, memory=None, next_memory=None):
        phi,_,_ = self.feature_net(obs, memory)
        predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        forward_error = (predicted_next_obs - next_obs).pow(2).mean()
        return forward_error


class TransitionLatentModel(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, feature_net) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim, feature_net)

        self.forward_dynamic_net = mlp(z_dim + action_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', z_dim)
        #self.apply(weight_init)
        

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, memory=None, next_memory=None):
        phi,_,_ = self.feature_net(obs,memory)
        with torch.no_grad():
            next_phi,_,_ = self.target_feature_net(next_obs,memory)
        predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        forward_error = (predicted_next_obs - next_phi.detach()).pow(2).mean()

        return forward_error


class AutoEncoder(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, feature_net) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim, feature_net)

        self.decoder = mlp(z_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', obs_dim)
        #self.apply(weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, memory=None, next_memory=None):
        del next_obs
        del action
        phi,_,_ = self.feature_net(obs,memory)
        predicted_obs = self.decoder(phi)
        reconstruction_error = (predicted_obs - obs).pow(2).mean()
        return reconstruction_error
    

class ImageAutoEncoder(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, feature_net) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim, feature_net)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, (4, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, (2, 2))
        )
        ##self.apply(weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, memory=None, next_memory=None):
        del next_obs
        del action
        phi,_,_ = self.feature_net(obs, memory)
        predicted_obs = self.decoder(phi.reshape(-1,64,1,1)).transpose(3, 2).transpose(1, 3)
        reconstruction_error = (predicted_obs - obs.image).pow(2).mean()
        return reconstruction_error




# class SVDP(FeatureLearner):
#     def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, feature_net) -> None:
#         super().__init__(obs_dim, action_dim, z_dim, hidden_dim, feature_net)
#         self.mu_net = mlp(obs_dim + action_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
#         #self.apply(weight_init)

#     def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, memory=None, next_memory=None):
#         phi,_,_ = self.feature_net(next_obs, memory)
#         mu = self.mu_net(torch.cat([obs, action], dim=1))
#         P = torch.einsum("sd, td -> st", mu, phi)
#         I = torch.eye(*P.size(), device=P.device)
#         off_diag = ~I.bool()
#         loss = - 2 * P.diag().mean() + P[off_diag].pow(2).mean()

#         # orthonormality loss
#         Cov = torch.matmul(phi, phi.T)
#         I = torch.eye(*Cov.size(), device=Cov.device)
#         off_diag = ~I.bool()
#         orth_loss_diag = - 2 * Cov.diag().mean()
#         orth_loss_offdiag = Cov[off_diag].pow(2).mean()
#         orth_loss = orth_loss_offdiag + orth_loss_diag
#         loss += orth_loss

#         return loss


