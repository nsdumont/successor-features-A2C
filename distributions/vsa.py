import torch
from torch import nan
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.categorical import Categorical

import sys,os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from spaces import SSPDiscrete
    

def SPDistribution(mu, space):
    sims = (torch.from_numpy(space.ssp_space.vectors[None,:,:]).to(mu.device) * mu[:,None,:]).sum(axis = -1)
    probs = sims**2   
    probs = probs/probs.sum(axis=-1,keepdim=True)
    return Categorical(probs=probs)



class SSPDistribution(Distribution):

    def __init__(self, mu, space, validate_args=None):
        self.mu = mu
        self.space = space
        self._param = self.mu 
        self._num_events = self._param.size()[-1]
        batch_shape = (
            self._param.size()[:-1] if self._param.ndimension() > 1 else torch.Size()
        )
        super().__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            if not isinstance(sample_shape, torch.Size):
                sample_shape = torch.Size(sample_shape)
            mu_2d = self.mu.reshape(-1, self._num_events)
            samples_2d = torch.from_numpy(self.space.ssp_space.decode(mu_2d.numpy())).to(self.mu.device)
            return samples_2d.reshape(self._extended_shape(sample_shape))
    
      
    def prob(self, value):
        mu_2d = self.mu.reshape(-1, self._num_events)
        return (value * mu_2d).sum(axis = -1)

    def log_prob(self, value):
        mu_2d = self.mu.reshape(-1, self._num_events)
        sims = (value * mu_2d).sum(axis = -1)
        return 2*torch.log(torch.abs(sims))

    def entropy(self):
        with torch.no_grad():
            samples = torch.from_numpy(self.space.samples(100)).to(self.mu.device)
        log_probs = self.log_prob(samples)
        probs = self.prob(samples)
        p_log_p = probs*log_probs
        return -p_log_p.mean(-1)
