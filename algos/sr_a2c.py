import numpy as np
import torch
import torch.nn.functional as F
import itertools
from torch.autograd import Variable
from utils import DictList

from algos.baseSR import BaseSRAlgo


class SRAlgo(BaseSRAlgo):

    def __init__(self, envs, model,target, feature_learn="curiosity", device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, sr_loss_coef=1, policy_loss_coef=1,recon_loss_coef=1,reward_loss_coef=1,norm_loss_coef=1,
                 max_grad_norm=10, recurrence=1,rmsprop_alpha=0.99, rmsprop_eps=1e-8,memory_cap=200,batch_size=300, preprocess_obss=None, reshape_reward=None):
 
        num_frames_per_proc = num_frames_per_proc or 200

        super().__init__(envs, model, target, device, num_frames_per_proc, discount, lr, gae_lambda, 
                         max_grad_norm, recurrence, memory_cap, preprocess_obss, reshape_reward)

        self.norm_loss_coef = norm_loss_coef
        self.entropy_coef = entropy_coef
        self.sr_loss_coef = sr_loss_coef
        self.policy_loss_coef = policy_loss_coef
        self.recon_loss_coef = recon_loss_coef
        self.reward_loss_coef = reward_loss_coef
        self.feature_learn = feature_learn
        
        self.batch_size=300
        
        #params = [self.model.feature_in.parameters(), self.model.feature_out.parameters(), self.model.actor.parameters()]
        #self.feature_params = itertools.chain(*params)
        
        #self.feature_optimizer = torch.optim.RMSprop(self.feature_params, lr,alpha=rmsprop_alpha, eps=rmsprop_eps)
        self.feature_optimizer = torch.optim.RMSprop([{'params': self.model.feature_in.parameters()},{'params': self.model.feature_out.parameters()},
                                                      {'params': self.model.actor.parameters()}],
                                                     lr,alpha=rmsprop_alpha, eps=rmsprop_eps)
        self.actor_optimizer = torch.optim.RMSprop(self.model.actor.parameters(),
                                          lr,alpha=rmsprop_alpha, eps=rmsprop_eps, weight_decay=0.0)
        
        self.sr_optimizer = torch.optim.RMSprop(self.model.SR.parameters(),
                                          lr,alpha=rmsprop_alpha, eps=rmsprop_eps, weight_decay=0.0)
          
        self.reward_optimizer = torch.optim.RMSprop(self.model.reward.parameters(),
                                          lr,alpha=rmsprop_alpha, eps=rmsprop_eps)
        
        self.num_updates = 0
        

    def update_parameters(self, exps):
        # Compute starting indexes
        
        #with torch.autograd.set_detect_anomaly(True):

        inds = self._get_starting_indexes()

        # Initialize update values
        update_value_loss = 0
        update_entropy = 0
        update_policy_loss = 0
        update_reconstruction_loss = 0
        update_reward_loss = 0
        update_sr_loss = 0
        update_norm_loss = 0
        update_actor_loss = 0
        update_feature_loss = 0
        update_A_loss = 0

        # Initialize memory

        if self.model.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Run model
            #if self.model.feature_learn=="curiosity":
            if self.model.recurrent:
                _, _, _, predictions, _, _, _ = self.model(sb[:-1].obs, sb[:-1].action, sb[1:].obs, memory[:-1,:] * sb.mask[:-1])
            else:
                _, _, _, predictions, _, _ = self.model(sb[:-1].obs,sb[:-1].action,sb[1:].obs)
            # else:
            #     if self.model.recurrent:
            #         _, _, _, predictions, _, _, _ = self.model(sb.obs, sb.action, sb.obs, memory * sb.mask)
            #     else:
            #         _, _, _, predictions, _, _ = self.model(sb.obs,sb.action,sb.obs)
            if self.model.recurrent:
                dist, value, embedding, _, successor, reward, memory = self.model(sb.obs,memory= memory * sb.mask)
            else:
                dist, value, embedding, _, successor, reward = self.model(sb.obs)
                    
            # Compute loss
            
            # Feature loss
            if self.feature_learn == "reconstruction":
                reconstruction_loss = F.mse_loss(predictions, sb.obs.image)
            elif self.feature_learn=="curiosity":
                next_embedding, next_obs_pred, action_pred = predictions
                forward_loss = F.mse_loss(next_obs_pred , next_embedding)
                inverse_loss = F.nll_loss(action_pred, sb[:-1].action.long()) # mse if continuous action
                reconstruction_loss = forward_loss + inverse_loss 


            norm_loss = (torch.norm(embedding, dim=1) - 1).pow(2).mean()
            feature_loss = reconstruction_loss + self.norm_loss_coef*norm_loss 
            #reward_loss = F.mse_loss(reward, sb.reward )
            sr_loss = F.smooth_l1_loss(successor, sb.successorn) #F.mse_loss(successor, sb.successorn)
            entropy = dist.entropy().mean()
            
            with torch.no_grad():
                SR_advanage_dot_R = self.target.reward(sb.SR_advantage).reshape(-1) #modle or target 
                value_loss = (value - sb.returnn).pow(2).mean() # not used for optimization, just for logs

            
            A_diff = F.mse_loss(SR_advanage_dot_R, sb.V_advantage)
            #if self.num_updates < -1:
            #    policy_loss = -(dist.log_prob(sb.action) * sb.V_advantage).mean()
            #else:
            policy_loss = -(dist.log_prob(sb.action) * SR_advanage_dot_R).mean()
            actor_loss = policy_loss - self.entropy_coef * entropy 
        
            # Update batch values
            update_entropy += entropy.item()
            update_policy_loss += policy_loss.item()
            update_reconstruction_loss += reconstruction_loss.item()
            #update_reward_loss = update_reward_loss + reward_loss
            update_norm_loss += norm_loss.item()
            update_sr_loss = update_sr_loss + sr_loss
            update_actor_loss = update_actor_loss + actor_loss
            update_feature_loss = update_feature_loss + feature_loss
            update_value_loss += value_loss.item()
            update_A_loss += A_diff.item()

        # Update update values
        update_entropy /= self.recurrence
        update_value_loss /= self.recurrence
        update_policy_loss /= self.recurrence
        update_reconstruction_loss /= self.recurrence
        update_norm_loss /= self.recurrence
        #update_reward_loss = update_reward_loss/self.recurrence
        update_sr_loss = update_sr_loss/self.recurrence
        update_actor_loss = update_actor_loss/self.recurrence
        update_feature_loss = update_feature_loss/self.recurrence
        update_A_loss /= self.recurrence

        # Update actor-critic
        
        
        
        self.model.zero_grad()
        update_sr_loss.backward(retain_graph=True)
        update_grad_norm_sr = sum(p.grad.data.norm(2) ** 2 for p in self.model.SR.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.model.SR.parameters(), self.max_grad_norm)
        self.sr_optimizer.step()
        
        # reward leanring: not on policy so do random samples
        transitions = self.replay_memory.sample(np.min([self.batch_size,self.replay_memory.__len__()]))
        batch_state_t, batch_reward = zip(*transitions)
        batch_state = DictList()
        batch_state.image =  torch.cat(batch_state_t)
        batch_reward = torch.cat(batch_reward)
        if self.model.recurrent:
            _, _, _, _, _, reward, _ = self.model(batch_state) # issue with memory here
        else:
            _, _, _, _, _, reward = self.model(batch_state)
        update_reward_loss = F.smooth_l1_loss(reward, batch_reward.squeeze())
    
        self.model.zero_grad()
        update_reward_loss.backward(retain_graph=True)
        update_grad_norm_reward = sum(p.grad.data.norm(2) ** 2 for p in self.model.reward.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.model.reward.parameters(), self.max_grad_norm)
        self.reward_optimizer.step()
        
        # self.model.zero_grad()
        # update_actor_loss.backward(retain_graph=True)
        # update_grad_norm_actor = sum(p.grad.data.norm(2) ** 2 for p in self.model.actor.parameters()) ** 0.5
        # torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.max_grad_norm)
        # self.actor_optimizer.step()
        
        self.model.zero_grad()
        update_loss = self.recon_loss_coef*update_feature_loss + update_actor_loss
        update_loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.model.feature_in.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.model.feature_out.parameters(), self.max_grad_norm)
        self.feature_optimizer.step()
        
        
        update_grad_norm = np.max([ update_grad_norm_reward.item(),update_grad_norm_sr.item()]) #update_grad_norm_sr.item()


        # Log some values

        logs = {
            "reconstruction_loss": update_reconstruction_loss,
            "reward_loss": update_reward_loss.item(),
            "sr_loss": update_sr_loss.item(),
            "norm_loss": update_norm_loss,
            "entropy": update_entropy,
            "value_loss": update_value_loss,
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


