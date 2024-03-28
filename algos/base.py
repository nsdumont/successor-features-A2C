from abc import ABC, abstractmethod
import torch
import numpy as np
# from torch_ac.format import default_preprocess_obss
# from torch_ac.utils import DictList, ParallelEnv

from utils import default_preprocess_obss, DictList, ParallelEnv
# from algos.blr import BayesianLinearRegression 
from gymnasium.spaces.dict import Dict

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, model, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,entropy_decay,
                 value_loss_coef, dissim_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        model : torch.Module
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
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.entropy_decay=entropy_decay
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.continuous_action = model.continuous_action
        self.dissim_coef=dissim_coef

        # Control parameters

        assert self.model.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure model

        self.model.to(self.device)
        self.model.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        self.preprocessed_obss = [None]*(shape[0])
        if self.model.recurrent:
            self.memory = torch.zeros(shape[1], self.model.memory_size, device=self.device)
            self.target_memory = torch.zeros(shape[1], self.model.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.model.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        if self.continuous_action:
            self.actions = torch.zeros(self.num_frames_per_proc, self.num_procs, self.model.n_actions, device=self.device)
        else:
            self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
        if dissim_coef>0:
            if type(envs[0].observation_space)==Dict:
                self.obs_mean =  torch.zeros( envs[0].observation_space['image'].shape_out,   device=self.device)
            else:
                self.obs_mean =  torch.zeros(envs[0].observation_space.shape_out, device=self.device)
        
        #     if type(envs[0].observation_space)==Dict:
        #         self.sigma =  torch.zeros(shape[1], envs[0].observation_space['image'].shape_out,  envs[0].observation_space['image'].shape_out, device=self.device)
        #     else:
        #         self.sigma =  torch.zeros(shape[1], envs[0].observation_space.shape_out,  envs[0].observation_space.shape_out, device=self.device)
        
        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs
        
        
        # self.use_blr=use_blr
        # if use_blr:
        #     self.blr = BayesianLinearRegression(envs[0].observation_space['image'].shape_out)

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
                self.obs = [o for o in self.model.scaler.transform(self.obs)]

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.model.recurrent:
                    dist, value, memory = self.model(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.model(preprocessed_obs)
            
            if self.continuous_action:
                action = dist.sample().detach()
                action = torch.clamp(action, self.env.envs[0].min_action, self.env.envs[0].max_action)
                torch.nan_to_num(action, nan=torch.Tensor(self.env.envs[0].action_space.sample()), posinf=self.env.envs[0].max_action, neginf=self.env.envs[0].min_action)
            else:
                action = dist.sample().detach()

            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = tuple(a | b for a, b in zip(terminated, truncated))

            # Update experiences values

            self.obss[i] = self.obs
            if self.dissim_coef > 0:
                self.obs_mean += preprocessed_obs.image.sum(axis=0) #old version: no discount
                self.obs_mean /= torch.linalg.norm(self.obs_mean)
            # self.sigma += self.discount * torch.bmm(preprocessed_obs.image.unsqueeze(2), preprocessed_obs.image.unsqueeze(1))
            self.preprocessed_obss[i] = preprocessed_obs
            self.obs = obs
            if self.model.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action).squeeze()

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
            self.obs = [o for o in self.model.scaler.transform(self.obs)]

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.model.recurrent:
                _, next_value, _ = self.model(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.model(preprocessed_obs)

        if self.dissim_coef > 0:
            vec_obs = torch.stack([torch.stack([self.preprocessed_obss[i][j].image
                        for j in range(self.num_procs) ]) for i in range(self.num_frames_per_proc)])
            # vec_obs = torch.stack([torch.stack([self.preprocessed_obss[j][i].image
            #             for j in range(self.num_frames_per_proc) ]) for i in range(self.num_procs)])
            
            # self.intrinsic_rewards = torch.diagonal(torch.bmm(vec_obs, torch.bmm(self.sigma, torch.swapaxes(vec_obs,1,2))), dim1=1,dim2=2)
            # mu=vec_obs.sum(axis=(0,1))
            #mu /= torch.linalg.norm(mu)
            #sims = (vec_obs @ mu)**2
            sims = vec_obs @ (self.obs_mean/torch.linalg.norm(self.obs_mean))  #old version: self.obs_mean \= self.norm; sims are squared
            self.intrinsic_rewards = self.dissim_coef*(1-sims)
        else:
            self.intrinsic_rewards = torch.zeros(self.rewards.shape, device=self.device)
        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.intrinsic_rewards[i] + self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

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
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        
        # mu = exps.obs.image.sum(axis=0)
        # # mu = mu/torch.linalg.norm(mu)
        # sims = (exps.obs.image @ mu)**2
        # exps.sims = sims/sims.sum()
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
