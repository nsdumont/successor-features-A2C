import numpy as np
import torch
import torch.nn.functional as F
import itertools

from algos.baseSR import BaseSRAlgo

import sys,os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from utils import soft_update_params

class SRAlgo(BaseSRAlgo):

    def __init__(self, envs, model, feature_learn="curiosity", device=None, num_frames_per_proc=None, discount=0.99,  lr_feature=0.01,
        lr_actor = 0.01,lr_sr=0.01, lr_reward= 0.01/30, gae_lambda=0.95, dissim_coef=0.0, entropy_coef=0.01, entropy_decay=0, 
        max_grad_norm=10, recurrence=1,rmsprop_alpha=0.99, rmsprop_eps=1e-8,memory_cap=100000,batch_size=200, preprocess_obss=None, reshape_reward=None):
 
        num_frames_per_proc = num_frames_per_proc or 10

        super().__init__(envs, model, device, num_frames_per_proc, discount,  lr_feature, gae_lambda, dissim_coef, max_grad_norm, recurrence, memory_cap, preprocess_obss, reshape_reward)
      
        self.entropy_coef = entropy_coef
        self.entropy_decay = entropy_decay
        self.feature_learn = feature_learn
        self.batch_size=batch_size
        
        
        # if self.feature_learn == "combined":
        #     self.optimizer = torch.optim.RMSprop(self.model.parameters(),
        #                                   lr_sr,alpha=rmsprop_alpha, eps=rmsprop_eps)
        # else:
        if self.feature_learn != "none":
            self.feature_optimizer = torch.optim.RMSprop(list(self.model.feature_net.parameters()) +
                                                          list(self.model.feature_learner.parameters()) ,#{'params': self.model.actor.parameters()} ],
                                                          lr_feature,alpha=rmsprop_alpha, eps=rmsprop_eps)
            self.actor_optimizer = torch.optim.RMSprop(self.model.actor.parameters(),
                                              lr_actor,alpha=rmsprop_alpha, eps=rmsprop_eps)
        else:
            self.actor_optimizer = torch.optim.RMSprop(list(self.model.actor.parameters()) +
                                                       list(self.model.feature_net.parameters()),
                                              lr_actor,alpha=rmsprop_alpha, eps=rmsprop_eps)
        
        self.sr_optimizer = torch.optim.RMSprop(self.model.SR.parameters(),
                                          lr_sr,alpha=rmsprop_alpha, eps=rmsprop_eps)
        self.reward_optimizer = torch.optim.RMSprop(self.model.reward.parameters(),
                                      lr_reward,alpha=rmsprop_alpha, eps=rmsprop_eps)
        
        self.num_updates = 0
        

    def update_parameters(self, exps):
        # Compute starting indexes
        
        #with torch.autograd.set_detect_anomaly(True):

        inds = self._get_starting_indexes()

        # Initialize update values
        update_entropy = 0
        update_policy_loss = 0
        update_sr_loss = 0#torch.zeros(1, requires_grad=True, device=self.device)
        update_actor_loss = 0#torch.zeros(1, requires_grad=True, device=self.device)
        update_feature_loss = 0#torch.zeros(1, requires_grad=True, device=self.device)
        update_reward_loss = 0
        # update_A_loss= 0
        # Initialize memory

        if self.model.use_memory:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Run model
            if not self.model.continuous_action:
                processed_action = F.one_hot(sb[:-1].action.long(),self.model.n_actions)
                #sb[:-1].action.long()#
            else:
                processed_action = sb[:-1].action
            if self.model.use_memory:
                _, _, _, feature_loss, _, _, _ = self.model(sb[:-1].obs,processed_action, sb[1:].obs, memory[:-1,:] * sb.mask[:-1])
            else:
                _, _, _, feature_loss, _, _, _ = self.model(sb[:-1].obs, processed_action,sb[1:].obs)
     
            if self.model.use_memory:
                dist, value, embedding, _, successor, reward, memory = self.model(sb.obs, memory= memory * sb.mask)
            else:
                dist, value, embedding, _, successor, reward, _ = self.model(sb.obs)
                    
            # Compute loss

            reward_loss = F.mse_loss(reward, sb.reward )
            sr_loss = F.mse_loss(successor, sb.successorn) 
            entropy = dist.entropy().mean()
            
            
            # SR_advanage_dot_R = self.model.reward(sb.SR_advantage).detach()
            # A_diff = F.mse_loss(SR_advanage_dot_R, sb.V_advantage)
            policy_loss = -(dist.log_prob(sb.action) * sb.V_advantage).mean()
            #policy_loss = -(dist.log_prob(sb.action) * SR_advanage_dot_R).mean()
            actor_loss = policy_loss - self.entropy_coef * entropy
        
            # Update batch values
            update_entropy += entropy.item()
            update_policy_loss += policy_loss.item()
            # update_norm_loss += norm_loss.item()
            update_sr_loss += sr_loss
            update_actor_loss += actor_loss
            update_feature_loss += feature_loss
            # update_A_loss += A_diff.item()
            update_reward_loss += reward_loss

        # Update update values
        update_entropy /= self.recurrence
        update_policy_loss /= self.recurrence
        update_sr_loss /= self.recurrence
        update_actor_loss /= self.recurrence
        update_feature_loss /= self.recurrence
        update_reward_loss /= self.recurrence

        # Update all parts
        # Update SR
        # if self.feature_learn == "combined":
        #     update_loss = update_sr_loss + update_feature_loss + update_reward_loss + update_actor_loss
        #     self.optimizer.zero_grad()
        #     update_loss.backward(retain_graph=False)
        #     update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.model.parameters()) ** 0.5
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        #     self.optimizer.step()   
        # else:
        self.sr_optimizer.zero_grad()
        update_sr_loss.backward(retain_graph=True)
        update_grad_norm_sr = sum(p.grad.data.norm(2) ** 2 for p in self.model.SR.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.model.SR.parameters(), self.max_grad_norm)
        self.sr_optimizer.step()
        
        # Update actor (policy loss + entropy)
        self.actor_optimizer.zero_grad()
        update_actor_loss.backward(retain_graph=True)
        update_grad_norm_actor = sum(p.grad.data.norm(2) ** 2 for p in self.model.actor.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        
        self.reward_optimizer.zero_grad()
        update_reward_loss.backward(retain_graph=self.feature_learn != "none")
        update_grad_norm_reward = sum(p.grad.data.norm(2) ** 2 for p in self.model.reward.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.model.reward.parameters(), self.max_grad_norm)
        self.reward_optimizer.step()
        
         
        # Update feature embedding net
        if self.feature_learn != "none":
            self.feature_optimizer.zero_grad()
            update_feature_loss.backward(retain_graph=False)
            # update_grad_norm_features = sum(p.grad.data.norm(2) ** 2 for p in self.model.feature_net.parameters()) ** 0.5 + sum(p.grad.data.norm(2) ** 2 for p in self.model.feature_learner.parameters()) ** 0.5
            torch.nn.utils.clip_grad_norm_(self.model.feature_net.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model.feature_learner.parameters(), self.max_grad_norm)
            self.feature_optimizer.step()
            
            soft_update_params(self.model.feature_net, self.model.target_feature_net, 0.01)
            
            
        update_grad_norm = np.max([ update_grad_norm_sr.item(),update_grad_norm_reward.item(),
                                       update_grad_norm_actor.item()]) 
            
        

        
        # Log some values
        self.entropy_coef = self.entropy_coef*(1-self.entropy_decay)
        logs = {
            "feature_loss": update_feature_loss.item(),
            "reward_loss": update_reward_loss.item(),
            "sr_loss": update_sr_loss.item(),
            "entropy": update_entropy,
            "policy_loss": update_policy_loss,
            "grad_norm": update_grad_norm # "A_mse": update_A_loss
        }
        
        self.num_updates += 1

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """
        starting_indexes = np.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
    
    def _get_feature_params(self):
        params = [self.model.feature_in.parameters(), self.model.feature_out.parameters(), self.model.actor.parameters()]
        return itertools.chain(*params)


