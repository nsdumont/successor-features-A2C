import numpy as np
import torch
import torch.nn.functional as F
import itertools
import torch.autograd
from torch.autograd import Variable
from utils import DictList

from algos.baseSR import BaseSRAlgo


class SRPPOAlgo(BaseSRAlgo):

    def __init__(self, envs, model,target, feature_learn="curiosity", device=None, num_frames_per_proc=None, discount=0.99,  lr_feature=0.01,
        lr_actor = 0.01,lr_sr=0.01, lr_reward= 0.01/30, gae_lambda=0.95, entropy_coef=0.01,  recon_loss_coef=1,norm_loss_coef=1,
        max_grad_norm=10, recurrence=1,rmsprop_alpha=0.99, rmsprop_eps=1e-8,memory_cap=200, epochs=4, batch_size=256, clip_eps=0.2, preprocess_obss=None, reshape_reward=None,use_V_advantage=False):
 
        num_frames_per_proc = num_frames_per_proc or 200

        super().__init__(envs, model, target, device, num_frames_per_proc, discount,  lr_feature, gae_lambda, max_grad_norm, recurrence, memory_cap, preprocess_obss, reshape_reward)
      
        #torch.autograd.set_detect_anomaly(True)
        
        self.norm_loss_coef = norm_loss_coef
        self.entropy_coef = entropy_coef
        self.recon_loss_coef = recon_loss_coef
        self.feature_learn = feature_learn
        self.clip_eps = clip_eps
        self.batch_size=batch_size
        self.use_V_advantage= use_V_advantage
        self.epochs = epochs
        #params = [self.model.feature_in.parameters(), self.model.feature_out.parameters(), self.model.actor.parameters()]
        #self.feature_params = itertools.chain(*params)
        
        #self.feature_optimizer = torch.optim.RMSprop(self.feature_params, lr,alpha=rmsprop_alpha, eps=rmsprop_eps)

        self.feature_optimizer = torch.optim.RMSprop([{'params': self.model.feature_in.parameters()},{'params': self.model.feature_out.parameters()},
                                                      {'params': self.model.actor.parameters()}],
                                                     lr_feature,alpha=rmsprop_alpha, eps=rmsprop_eps)
        #self.actor_optimizer = torch.optim.RMSprop(self.model.actor.parameters(),
         #                                 lr_actor,alpha=rmsprop_alpha, eps=rmsprop_eps, weight_decay=0.0)
        
        self.sr_optimizer = torch.optim.RMSprop(self.model.SR.parameters(),
                                          lr_sr,alpha=rmsprop_alpha, eps=rmsprop_eps, weight_decay=0.0)
          
        self.reward_optimizer = torch.optim.RMSprop([{'params': self.model.feature_in.parameters()},{'params': self.model.reward.parameters()}],
                                          lr_reward,alpha=rmsprop_alpha, eps=rmsprop_eps) #30
        
        #self.optimizer =  torch.optim.RMSprop(self.model.parameters(),
         #                                 lr_reward,alpha=rmsprop_alpha, eps=rmsprop_eps) 
        
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
    
                if self.model.recurrent:
                    memory = exps.memory[inds]
    
                for i in range(self.recurrence):
                    # Create a sub-batch of experience
    
                    sb = exps[inds + i]
                     
                    # Compute loss
                    if self.model.recurrent:
                        _, _, _, predictions, _, _, _,_ = self.model(sb[:-1].obs, sb[:-1].action, sb[1:].obs, memory[:-1,:] * sb.mask[:-1])
                    else:
                        _, _, _, predictions, _, _,_ = self.model(sb[:-1].obs,sb[:-1].action,sb[1:].obs)
                      
                    if self.model.recurrent:
                        dist, value, embedding, _, successor, _,_, memory = self.model(sb.obs,memory= memory * sb.mask)
                    else:
                        dist, value, embedding, _, successor, _,_ = self.model(sb.obs)
                             
                    if self.feature_learn == "reconstruction":
                        reconstruction_loss = F.mse_loss(predictions, sb.obs[:-1].image)
                    elif self.feature_learn=="curiosity":
                        next_embedding, next_obs_pred, action_pred = predictions
                        forward_loss = F.mse_loss(next_obs_pred , next_embedding)
                        if self.model.continuous_action:
                            inverse_loss = F.mse_loss(action_pred.reshape(-1),sb[:-1].action.float())
                        else:
                            inverse_loss = F.nll_loss(action_pred, sb[:-1].action.long()) # mse if continuous action
                        reconstruction_loss = forward_loss + inverse_loss 
    
                    norm_loss = (torch.norm(embedding, dim=1) - 1).pow(2).mean()
                    feature_loss = reconstruction_loss + self.norm_loss_coef*norm_loss 
                    
                    
                    
                    sr_loss = F.mse_loss(successor, sb.successorn).clamp(max=self.clip_eps)
         
                    # value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    # surr1 = (value - sb.returnn).pow(2)
                    # surr2 = (value_clipped - sb.returnn).pow(2)
                    # value_loss = torch.max(surr1, surr2).mean()
                    
                    with torch.no_grad():
                        _,_,_,_,_,_,r_vec,_ = self.target(sb.obs,memory= memory * sb.mask)
                        SR_advanage_dot_R = torch.sum(r_vec * sb.SR_advantage, 1)
    
    
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    if self.use_V_advantage:
                        surr1 = ratio * sb.V_advantage
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.V_advantage
                    else:
                        surr1 = ratio * SR_advanage_dot_R
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * SR_advanage_dot_R
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
    
                    if self.model.recurrent and i < self.recurrence - 1:
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
                update_grad_norm_sr = sum(p.grad.data.norm(2) ** 2 for p in self.model.SR.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.model.SR.parameters(), self.max_grad_norm)
                self.sr_optimizer.step()
                
                self.model.zero_grad()
                update_loss = self.recon_loss_coef*batch_feature_loss + batch_actor_loss
                update_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.feature_in.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model.feature_out.parameters(), self.max_grad_norm)
                self.feature_optimizer.step()
                
                # reward leanring: not on policy so do random samples
                transitions = self.replay_memory.sample(np.min([self.batch_size,self.replay_memory.__len__()]))
                batch_state_img, batch_state_txt, batch_reward = zip(*transitions)
                batch_state = DictList()
                batch_state.image =  torch.cat(batch_state_img)
                batch_state.text = torch.cat(batch_state_txt)
                batch_reward = torch.cat(batch_reward)
                if self.model.recurrent:
                    _, _, _, _, _, reward, _,_ = self.model(batch_state) # issue with memory here
                else:
                    _, _, _, _, _, reward,_ = self.model(batch_state)
                batch_reward_loss = F.smooth_l1_loss(reward, batch_reward.reshape(reward.shape))

                self.model.zero_grad()
                batch_reward_loss.backward(retain_graph=False)
                update_grad_norm_reward = sum(p.grad.data.norm(2) ** 2 for p in self.model.reward.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.model.reward.parameters(), self.max_grad_norm)
                self.reward_optimizer.step()
                
                grad_norm = np.max([ update_grad_norm_reward.item(),update_grad_norm_sr.item()]) #update_grad_norm_sr.item()

    
                # Update log values
    
                log_entropies.append(batch_entropy)
                log_feature_losses.append(batch_feature_loss.item())
                log_policy_losses.append(batch_policy_loss)
                log_reward_losses.append(batch_reward_loss.item())
                log_sr_losses.append(batch_sr_loss.item())
                log_grad_norms.append(grad_norm)
    
        # Log some values
    
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
        params = [self.model.feature_in.parameters(), self.model.feature_out.parameters(), self.model.actor.parameters()]
        return itertools.chain(*params)


