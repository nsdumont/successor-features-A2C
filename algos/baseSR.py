from abc import ABC, abstractmethod
import torch
import random
import math
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
from ReplayMemory import ReplayMemory


import numpy as np

class BaseSRAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, model, target, device, num_frames_per_proc, discount, lr, gae_lambda, 
                 max_grad_norm, recurrence, memory_cap, preprocess_obss, reshape_reward=None):
        """
        Initializes a `BaseSRAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters
        self.env = ParallelEnv(envs)
        self.model = model
        self.target = target
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        
        self.replay_memory = ReplayMemory(memory_cap)
        
        use_cuda = torch.cuda.is_available()
        self.FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        
        self.total_updates = 0
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.continuous_action = model.continuous_action

        # Control parameters

        assert self.model.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.model.to(self.device)
        self.model.train()
        self.target.to(self.device)
        self.target.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)
        vec_shape = (self.num_frames_per_proc, self.num_procs,self.model.embedding_size)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        if self.model.recurrent:
            self.memory = torch.zeros(shape[1], self.model.memory_size, device=self.device)
            self.target_memory = torch.zeros(shape[1], self.model.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.model.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        if self.continuous_action:
            self.actions = torch.zeros(*shape, device=self.device)
        else:
            self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.target_values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.SR_advantages = torch.zeros(*vec_shape, device=self.device)
        self.V_advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        self.embeddings = torch.zeros(*vec_shape, device=self.device)
        self.target_embeddings = torch.zeros(*vec_shape, device=self.device)
        self.successors = torch.zeros(*vec_shape, device=self.device)
        self.target_successors = torch.zeros(*vec_shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            
            if self.continuous_action:
                self.obs = [self.model.scaler.transform(self.obs[0].reshape(1, -1)).reshape(-1).astype('float64')]
          

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            
            with torch.no_grad():
                if self.model.recurrent:
                    dist, value, embedding, _, successor, _,_, memory = self.model(preprocessed_obs, memory=self.memory * self.mask.unsqueeze(1)) 
                    _, target_value, target_embedding, _, target_successor, _,_, _ = self.target(preprocessed_obs, memory=self.memory * self.mask.unsqueeze(1)) 
                   # target
                else:
                    dist, value, embedding, _, successor, _,_,_ = self.model(preprocessed_obs)
                    _, target_value, target_embedding, _, target_successor, _,_,_ = self.target(preprocessed_obs)
                    
                

            
            if self.continuous_action:
                # Should this (eps + stochastic policy) be done? Or use (eps + det policy) or just stochastic policy?
                # epsample = random.random()
                # eps_threshold = 0.02 + (0.9 - 0.02) * math.exp(-1. * self.total_updates / 200)
                # if epsample > eps_threshold:
                #     noise_dist = torch.distributions.normal.Normal(0, 0.03)
                action = dist.sample()# + noise_dist.sample()
                action = torch.clamp(action, self.env.envs[0].min_action, self.env.envs[0].max_action).reshape(-1)#.item()
                if torch.isnan(action):
                    action = torch.Tensor(self.env.envs[0].action_space.sample())
                # else:
                #     action = torch.Tensor(self.env.envs[0].action_space.sample())
               
                obs, reward, done, _ = self.env.step([action])
                
                obs = (obs[0].reshape(1,-1)) ##
                
                # try:
                #     if np.any(np.isnan(obs[0])):
                #         obs
                # except TypeError:
                #     obs
                #change
                self.replay_memory.push((preprocessed_obs.image, torch.zeros(1),
                                     self.FloatTensor([reward])))
            else:
                action = dist.sample()
                obs, reward, done, _ = self.env.step(action.cpu().numpy())
                
                self.replay_memory.push((preprocessed_obs.image, preprocessed_obs.text,
                                     self.FloatTensor([reward])))

            # Update experiences values
            #preprocessed_obs_new = self.preprocess_obss(obs, device=self.device)

            


            self.obss[i] = self.obs
            self.obs = obs
            if self.model.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            self.target_values[i] = target_value
            self.embeddings[i] = embedding
            self.target_embeddings[i] = target_embedding
            self.successors[i] = successor
            self.target_successors[i] = target_successor
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            if self.continuous_action:
                self.log_probs[i] = torch.exp(-0.5*(action - dist.mean).pow(2)/dist.stddev.pow(2))/(dist.stddev*2*np.pi)

            else:
                self.log_probs[i] = dist.log_prob(action)

##TODO:
    #for continuous actions need to collect mean and var as well so that they can be used in ppo ratio


            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences
        
        if self.continuous_action:
            # asuming flat observations for continuous action case:
                # this is true for the Mountain Cart example but may not be in general
                # Ideally the continuous action code should be modifed to handle flat or image input
                # And the use of a scaler should be an option to train.py
                # And either use checks here to do the following
                # or create a wrapper that does the scaling and set it up in train.py
            self.obs[0]= self.model.scaler.transform(self.obs[0].reshape(1, -1)).reshape(-1)
            self.obs = self.obs.astype('float32')

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.model.recurrent:
                _, next_value, _, _, next_successor, _, _,_ = self.target(preprocessed_obs, memory=self.memory * self.mask.unsqueeze(1)) #target
            else:
                _, next_value, _, _, next_successor, _,_ = self.target(preprocessed_obs)


        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_successor = self.target_successors[i+1] if i < self.num_frames_per_proc - 1 else next_successor
            next_value = self.target_values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_SR_advantage = self.SR_advantages[i+1] if i < self.num_frames_per_proc - 1 else 0
            next_V_advantage = self.V_advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            SR_delta = self.target_embeddings[i] + (self.discount * next_successor * next_mask.reshape(-1,1)) - self.successors[i]
            self.SR_advantages[i] = SR_delta + (self.discount * self.gae_lambda * next_SR_advantage * next_mask.reshape(-1,1))
            
            V_delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.V_advantages[i] = V_delta + self.discount * self.gae_lambda * next_V_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.model.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.SR_advantage = self.SR_advantages.transpose(0, 1).reshape(-1,self.model.embedding_size)
        exps.successor = self.successors.transpose(0, 1).reshape(-1,self.model.embedding_size)
        exps.successorn = exps.successor + exps.SR_advantage
        exps.V_advantage = self.V_advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.V_advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
