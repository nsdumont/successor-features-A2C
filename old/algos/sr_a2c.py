import numpy as np
import torch
import torch.nn.functional as F
import itertools
from torch.autograd import Variable
from utils import DictList

from algos.baseSR import BaseSRAlgo


class SRAlgo(BaseSRAlgo):

    def __init__(self, envs, model,target, feature_learn="curiosity", device=None, num_frames_per_proc=None, discount=0.99,  lr_feature=0.01,
        lr_actor = 0.01,lr_sr=0.01, lr_reward= 0.01/30, gae_lambda=0.95, entropy_coef=0.01, recon_loss_coef=1,norm_loss_coef=1,
        max_grad_norm=10, recurrence=1,rmsprop_alpha=0.99, rmsprop_eps=1e-8,memory_cap=100000,batch_size=200, preprocess_obss=None, reshape_reward=None):
 
        num_frames_per_proc = num_frames_per_proc or 10

        super().__init__(envs, model, target, device, num_frames_per_proc, discount,  lr_feature, gae_lambda, max_grad_norm, recurrence, memory_cap, preprocess_obss, reshape_reward)
      
        self.norm_loss_coef = norm_loss_coef
        self.entropy_coef = entropy_coef
        self.recon_loss_coef = recon_loss_coef
        self.feature_learn = feature_learn
        self.batch_size=batch_size
        
        if self.feature_learn != "none":
            self.feature_optimizer = torch.optim.RMSprop(list(self.model.feature_in.parameters()) +
                                                          list(self.model.feature_out.parameters()) ,#{'params': self.model.actor.parameters()} ],
                                                          lr_feature,alpha=rmsprop_alpha, eps=rmsprop_eps)
        self.actor_optimizer = torch.optim.RMSprop(self.model.actor.parameters(),
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
        update_reconstruction_loss = 0
        update_sr_loss = 0#torch.zeros(1, requires_grad=True, device=self.device)
        update_norm_loss = 0
        update_actor_loss = 0#torch.zeros(1, requires_grad=True, device=self.device)
        update_feature_loss = 0#torch.zeros(1, requires_grad=True, device=self.device)
        update_reward_loss = 0
        update_A_loss= 0
        # Initialize memory

        if self.model.use_memory:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Run model
            if self.model.use_memory:
                _, _, _, predictions, _, _, _ = self.model(sb[:-1].obs, sb[:-1].action, sb[1:].obs, memory[:-1,:] * sb.mask[:-1])
            else:
                _, _, _, predictions, _, _, _ = self.model(sb[:-1].obs,sb[:-1].action,sb[1:].obs)
     
            if self.model.use_memory:
                dist, value, embedding, _, successor, _, memory = self.model(sb.obs, memory= memory * sb.mask)
            else:
                dist, value, embedding, _, successor, _, _ = self.model(sb.obs)
                    
            # Compute loss
            
            # Feature loss
            if self.feature_learn == "reconstruction":
                reconstruction_loss = F.mse_loss(predictions, sb.obs[:-1].image)
            elif self.feature_learn=="curiosity":
                next_embedding, next_obs_pred, action_pred = predictions
                forward_loss = F.mse_loss(next_obs_pred, next_embedding)
                if self.model.continuous_action:
                    inverse_loss = F.mse_loss(action_pred.reshape(-1),sb[:-1].action.float())
                else:
                    inverse_loss = F.nll_loss(action_pred, sb[:-1].action.long()) # mse if continuous action
                reconstruction_loss = forward_loss + inverse_loss 
            
            if self.feature_learn != "none":
                norm_loss = (torch.norm(embedding, dim=1) - 1).pow(2).mean()
                feature_loss = self.recon_loss_coef*reconstruction_loss + self.norm_loss_coef*norm_loss 
            else:
                reconstruction_loss = torch.Tensor(np.zeros(1))
                norm_loss = torch.Tensor(np.zeros(1))
                feature_loss = torch.Tensor(np.zeros(1))

            
            # reward_loss = F.mse_loss(reward, sb.reward )
            sr_loss = F.mse_loss(successor, sb.successorn) 
            entropy = dist.entropy().mean()
            
            
            SR_advanage_dot_R = self.model.reward(sb.SR_advantage).detach()
            A_diff = F.mse_loss(SR_advanage_dot_R, sb.V_advantage)
            policy_loss = -(dist.log_prob(sb.action) * sb.V_advantage).mean()
            #policy_loss = -(dist.log_prob(sb.action) * SR_advanage_dot_R).mean()
            actor_loss = policy_loss - self.entropy_coef * entropy
        
            # Update batch values
            update_entropy += entropy.item()
            update_policy_loss += policy_loss.item()
            update_reconstruction_loss += reconstruction_loss.item()
            update_norm_loss += norm_loss.item()
            update_sr_loss += sr_loss
            update_actor_loss += actor_loss
            update_feature_loss += feature_loss
            update_A_loss += A_diff.item()
            # update_reward_loss += reward_loss

        # Update update values
        update_entropy /= self.recurrence
        update_policy_loss /= self.recurrence
        update_reconstruction_loss /= self.recurrence
        update_norm_loss /= self.recurrence
        update_sr_loss /= self.recurrence
        update_actor_loss /= self.recurrence
        update_feature_loss /= self.recurrence
        # update_reward_loss /= self.recurrence

        # Update all parts
        # Update SR
        self.sr_optimizer.zero_grad()
        update_sr_loss.backward(retain_graph=True)
        update_grad_norm_sr = sum(p.grad.data.norm(2) ** 2 for p in self.model.SR.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.model.SR.parameters(), self.max_grad_norm)
        self.sr_optimizer.step()
        
        # Update actor (policy loss + entropy)
        self.actor_optimizer.zero_grad()
        update_actor_loss.backward(retain_graph=self.feature_learn != "none")
        update_grad_norm_actor = sum(p.grad.data.norm(2) ** 2 for p in self.model.actor.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        
         
        # Update feature embedding net
        if self.feature_learn != "none":
            self.feature_optimizer.zero_grad()
            #update_loss = update_feature_loss + update_reward_loss
            update_feature_loss.backward(retain_graph=False)
            update_grad_norm_features = sum(p.grad.data.norm(2) ** 2 for p in self.model.feature_in.parameters()) ** 0.5
            torch.nn.utils.clip_grad_norm_(self.model.feature_in.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model.feature_out.parameters(), self.max_grad_norm)
            self.feature_optimizer.step()
        else:
            update_grad_norm_features = torch.Tensor(np.zeros(1))
        
        
        # Update linear reward function: not on policy so do random samples
        transitions = self.replay_memory.sample(np.min([self.batch_size,self.replay_memory.__len__()]))
        batch_state_img, batch_state_txt, batch_reward = zip(*transitions)
        batch_state = DictList()
        batch_state.image =  torch.cat(batch_state_img)
        batch_state.text = torch.cat(batch_state_txt)
        batch_reward = torch.cat(batch_reward)
        _, _, _, _, _, reward, _ = self.model(batch_state) 
        update_reward_loss = F.mse_loss(reward, batch_reward.reshape(reward.shape)) #smooth_l1_loss,mse_loss
        self.reward_optimizer.zero_grad()
        update_reward_loss.backward(retain_graph=False)
        update_grad_norm_reward = sum(p.grad.data.norm(2) ** 2 for p in self.model.reward.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.model.reward.parameters(), self.max_grad_norm)
        self.reward_optimizer.step()
        
        
        update_grad_norm = np.max([ update_grad_norm_sr.item(),update_grad_norm_reward.item(),
                                   update_grad_norm_actor.item(), 
                                   update_grad_norm_features.item()]) 

        
        
        # Log some values

        logs = {
            "feature_loss": update_feature_loss.item(),
            "reward_loss": update_reward_loss.item(),
            "sr_loss": update_sr_loss.item(),
            "norm_loss": update_norm_loss,
            "entropy": update_entropy,
            "policy_loss": update_policy_loss,
            "grad_norm": update_grad_norm,
            "A_mse": update_A_loss
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


