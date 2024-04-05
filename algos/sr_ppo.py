import numpy as np
import torch
import torch.nn.functional as F
import itertools
import torch.autograd
from utils import DictList

from algos.baseSR import BaseSRAlgo

import sys,os
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))
from utils import soft_update_params

class SRPPOAlgo(BaseSRAlgo):

    def __init__(self, envs, model, feature_learn="cm", device=None, num_frames_per_proc=None, discount=0.99,  lr_feature=0.01,
        lr_actor = 0.01,lr_sr=0.01, lr_reward= 0.01/30, gae_lambda=0.95, dissim_coef=0.01, entropy_coef=0.01, entropy_decay=0, recon_loss_coef=1,
        max_grad_norm=10, recurrence=1,rmsprop_alpha=0.99, rmsprop_eps=1e-8,memory_cap=200, epochs=4, batch_size=256, clip_eps=0.2, preprocess_obss=None, reshape_reward=None,use_V_advantage=True):
 
        num_frames_per_proc = num_frames_per_proc or 200

        super().__init__(envs, model, device, num_frames_per_proc, discount,  lr_feature, gae_lambda, max_grad_norm, recurrence, memory_cap, preprocess_obss, reshape_reward)
      
        #torch.autograd.set_detect_anomaly(True)
        
        self.dissim_coef = dissim_coef
        self.entropy_coef = entropy_coef
        self.entropy_decay, = entropy_decay
        self.recon_loss_coef = recon_loss_coef
        self.feature_learn = feature_learn
        self.clip_eps = clip_eps
        self.batch_size=batch_size
        self.use_V_advantage= use_V_advantage
        self.epochs = epochs

        self.feature_optimizer = torch.optim.RMSprop([{'params': self.model.feature_net.parameters()},
                                                      {'params': self.model.feature_learner.parameters()} ],
                                                     lr_feature,alpha=rmsprop_alpha, eps=rmsprop_eps)
        self.actor_optimizer = torch.optim.RMSprop(self.model.actor.parameters(),
                                          lr_actor,alpha=rmsprop_alpha, eps=rmsprop_eps, weight_decay=0.0)
        
        self.sr_optimizer = torch.optim.RMSprop(self.model.SR.parameters(),
                                          lr_sr,alpha=rmsprop_alpha, eps=rmsprop_eps, weight_decay=0.0)
          
        self.reward_optimizer = torch.optim.RMSprop(self.model.reward.parameters(),
                                          lr_reward,alpha=rmsprop_alpha, eps=rmsprop_eps) #30
        

        self.batch_num = 0
 
    def update_parameters(self, exps):
        # Collect experiences
    
        for _ in range(self.epochs):
            # Initialize log values
    
            log_entropies = []
            log_feature_losses = []
            log_policy_losses = []
            log_reward_losses = []
            log_sr_losses = []
            log_grad_norms = []
    
            for inds in self._get_batches_starting_indexes():
                # Initialize batch values
    
                batch_entropy = 0
                batch_policy_loss = 0
                batch_sr_loss = 0
                batch_actor_loss = 0
                batch_feature_loss = 0
                # Initialize memory
    
                if self.model.use_memory:
                    memory = exps.memory[inds]
    
                for i in range(self.recurrence):
                    # Create a sub-batch of experience
    
                    sb = exps[inds + i]
                     
                    # Compute loss
                    if not self.model.continuous_action:
                        processed_action = F.one_hot(sb[:-1].action.long(),self.model.n_actions)
                    else:
                        processed_action = sb[:-1].action
                    if self.model.use_memory:
                        _, _, _, feature_loss, _, _, _ = self.model(sb[:-1].obs,processed_action, sb[1:].obs, memory[:-1,:] * sb.mask[:-1])
                    else:
                        _, _, _, feature_loss, _, _, _ = self.model(sb[:-1].obs, processed_action,sb[1:].obs)
             
                    if self.model.use_memory:
                        dist, value, embedding, _, successor, _,_, memory = self.model(sb.obs,memory= memory * sb.mask)
                    else:
                        dist, value, embedding, _, successor, _,_ = self.model(sb.obs)
                    
                    
                    
                    sr_clipped = sb.successor + torch.clamp(successor - sb.successor, -self.clip_eps, self.clip_eps)
                    surr1 = (successor - sb.successorn).pow(2)
                    surr2 = (sr_clipped - sb.successorn).pow(2)
                    sr_loss = torch.max(surr1, surr2).mean()
                    # sr_loss = F.mse_loss(successor, sb.successorn) 
                    
    
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.V_advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.V_advantage
                    
                    policy_loss = -torch.min(surr1, surr2).mean()

                    actor_loss = policy_loss - self.entropy_coef * entropy
                        
 
                    # Update batch values
                    batch_entropy += entropy.item()
                    batch_sr_loss += sr_loss
                    batch_policy_loss += policy_loss.item()
                    batch_actor_loss += actor_loss
                    batch_feature_loss += feature_loss
                    #batch_value_loss += value_loss.item()
    
                    # Update memories for next epoch
    
                    if self.model.use_memory and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()
    
                # Update batch values
    
                batch_entropy /= self.recurrence
                batch_sr_loss /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_actor_loss /= self.recurrence
                batch_feature_loss /= self.recurrence
                self.batch_num += 1
                # Update actor-critic
    
                
                
                self.sr_optimizer.zero_grad()
                batch_sr_loss.backward(retain_graph=True)
                update_grad_norm_sr = sum(p.grad.data.norm(2) ** 2 if p.requires_grad else 0 for p in self.model.SR.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.model.SR.parameters(), self.max_grad_norm)
                self.sr_optimizer.step()
                
                # Update actor (policy loss + entropy)
                self.actor_optimizer.zero_grad()
                batch_actor_loss.backward(retain_graph=True)
                update_grad_norm_actor = sum(p.grad.data.norm(2) ** 2 if p.requires_grad else 0 for p in self.model.actor.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                 
                # Update feature embedding net
                if self.feature_learn != "none":
                    self.feature_optimizer.zero_grad()
                    batch_feature_loss.backward(retain_graph=True)
                    update_grad_norm_features = sum(p.grad.data.norm(2) ** 2 if p.requires_grad else 0 for p in self.model.feature_net.parameters()) ** 0.5
                    torch.nn.utils.clip_grad_norm_(self.model.feature_net.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.model.feature_net.parameters(), self.max_grad_norm)
                    self.feature_optimizer.step()
                    
                    soft_update_params(self.model.feature_net, self.target_feature_net, 0.01)
                else:
                    update_grad_norm_features = torch.Tensor(np.zeros(1))
                
                # reward leanring: not on policy so do random samples
                transitions = self.replay_memory.sample(np.min([self.batch_size,self.replay_memory.__len__()]))
                batch_state_img, batch_state_txt, batch_reward = zip(*transitions)
                batch_state = DictList()
                batch_state.image =  torch.cat(batch_state_img)
                batch_state.text = torch.cat(batch_state_txt)
                batch_reward = torch.cat(batch_reward)
                if self.model.use_memory:
                    _, _, _, _, _, reward, _,_ = self.model(batch_state) # issue with memory here
                else:
                    _, _, _, _, _, reward,_ = self.model(batch_state)
                batch_reward_loss = F.smooth_l1_loss(reward, batch_reward.reshape(reward.shape))

                self.model.zero_grad()
                batch_reward_loss.backward(retain_graph=False)
                update_grad_norm_reward = sum(p.grad.data.norm(2) ** 2 if p.requires_grad else 0 for p in self.model.reward.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.model.reward.parameters(), self.max_grad_norm)
                self.reward_optimizer.step()
                
                grad_norm = np.max([ update_grad_norm_sr.item(),update_grad_norm_reward.item(),
                           update_grad_norm_actor.item(), 
                           update_grad_norm_features.item()]) 
    
                # Update log values
    
                log_entropies.append(batch_entropy)
                log_feature_losses.append(batch_feature_loss.item())
                log_policy_losses.append(batch_policy_loss)
                log_reward_losses.append(batch_reward_loss.item())
                log_sr_losses.append(batch_sr_loss.item())
                log_grad_norms.append(grad_norm)
    
        # Log some values
        self.entropy_coef = self.entropy_coef*(1-self.entropy_decay)
        logs = {
            "entropy": np.mean(log_entropies),
            "reward_loss": np.mean(log_reward_losses),
            "feature_loss": np.mean(log_feature_losses),
            "policy_loss": np.mean(log_policy_losses),
            "sr_loss": np.mean(log_sr_losses),
            "grad_norm": np.mean(log_grad_norms)
        }
    
        return logs


    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = np.arange(0, self.num_frames, self.recurrence)
        indexes = np.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
    
    def _get_feature_params(self):
        params = [self.model.feature_net.parameters(), self.model.feature_out.parameters(), self.model.actor.parameters()]
        return itertools.chain(*params)


