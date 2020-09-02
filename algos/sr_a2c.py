import numpy
import torch
import torch.nn.functional as F

from algos.baseSR import BaseSRAlgo

class SRAlgo(BaseSRAlgo):

    def __init__(self, envs, srmodel, feature_learn="curiosity", device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, sr_loss_coef=0.5, policy_loss_coef=1,recon_loss_coef=1,reward_loss_coef=0.5,
                 max_grad_norm=0.5, recurrence=4,rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None, 
                 norm_loss_coef=1, rank_loss_coef=0.01,continous_action=False):
        num_frames_per_proc = num_frames_per_proc or 60

        super().__init__(envs, srmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         sr_loss_coef,policy_loss_coef,recon_loss_coef,reward_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, continous_action)

        self.norm_loss_coef = norm_loss_coef
        self.rank_loss_coef = rank_loss_coef
        self.feature_learn = feature_learn
        
        self.feature_optimizer = torch.optim.RMSprop([{'params': self.srmodel.feature_in.parameters()},
                                                      {'params': self.srmodel.feature_out.parameters()},
                                                      {'params': self.srmodel.actor.parameters()}], 
                                                     lr, alpha=rmsprop_alpha, eps=rmsprop_eps)
        self.sr_optimizer = torch.optim.RMSprop(self.srmodel.SR.parameters(),
                                          lr,alpha=rmsprop_alpha, eps=rmsprop_eps)
          
        self.reward_optimizer = torch.optim.RMSprop(self.srmodel.reward.parameters(),
                                          lr,alpha=rmsprop_alpha, eps=rmsprop_eps)
        

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

        # Initialize memory

        if self.srmodel.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Run model
            if self.srmodel.recurrent:
                _, _, _, predictions, _, _, _ = self.srmodel(sb[:-1].obs, sb[:-1].action, sb[1:].obs, memory[:-1,:] * sb.mask[:-1])
                dist, value, embedding, _, successor, reward, memory = self.srmodel(sb.obs,memory= memory * sb.mask)
            else:
                _, _, _, predictions, _, _ = self.srmodel(sb[:-1].obs,sb[:-1].action,sb[1:].obs)
                dist, value, embedding, _, successor, reward = self.srmodel(sb.obs)
                    
            # Compute loss
            
            # Feature loss
            if self.feature_learn == "reconstruction":
                reconstruction_loss = F.mse_loss(predictions, sb.obs.image)
            elif self.feature_learn=="curiosity":
                next_embedding, next_obs_pred, action_pred = predictions
                forward_loss = F.mse_loss(next_obs_pred , next_embedding)
                inverse_loss = F.nll_loss(action_pred, sb[:-1].action.long()) 
                reconstruction_loss = forward_loss + inverse_loss 


            norm_loss = (torch.norm(embedding, dim=1) - 1).pow(2).mean()
            feature_loss = reconstruction_loss + self.norm_loss_coef*norm_loss 
            reward_loss = F.mse_loss(reward, sb.reward )
            sr_loss = F.mse_loss(successor, sb.successorn) + self.recon_loss_coef*feature_loss
            entropy = dist.entropy().mean()
            
            with torch.no_grad():
                SR_advanage_dot_R = self.srmodel.reward(sb.SR_advantage).reshape(-1)
                value_loss = (value - sb.returnn).pow(2).mean() # not used for optimization, just for logs

            policy_loss = -(dist.log_prob(sb.action) * SR_advanage_dot_R).mean()
            actor_loss = policy_loss - self.entropy_coef * entropy + self.recon_loss_coef*feature_loss
        
            # Update batch values
            update_entropy += entropy.item()
            update_policy_loss += policy_loss.item()
            update_reconstruction_loss += reconstruction_loss.item()
            update_reward_loss = update_reward_loss + reward_loss
            update_norm_loss += norm_loss.item()
            update_sr_loss = update_sr_loss + sr_loss
            update_actor_loss = update_actor_loss + actor_loss
            update_feature_loss = update_feature_loss + feature_loss
            update_value_loss += value_loss.item()

        # Update update values
        update_entropy /= self.recurrence
        update_value_loss /= self.recurrence
        update_policy_loss /= self.recurrence
        update_reconstruction_loss /= self.recurrence
        update_norm_loss /= self.recurrence
        update_reward_loss = update_reward_loss/self.recurrence
        update_sr_loss = update_sr_loss/self.recurrence
        update_actor_loss = update_actor_loss/self.recurrence
        update_feature_loss = update_feature_loss/self.recurrence

        # Update actor-critic
        update_grad_norm =0
        
        self.srmodel.zero_grad()
        update_sr_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.srmodel.SR.parameters(), self.max_grad_norm)
        self.sr_optimizer.step()
    
        self.srmodel.zero_grad()
        update_reward_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.srmodel.reward.parameters(), self.max_grad_norm)
        self.reward_optimizer.step()
        
        self.srmodel.zero_grad()
        update_actor_loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.srmodel.actor.parameters(), self.max_grad_norm)
        self.feature_optimizer.step()


        # Log some values

        logs = {
            "reconstruction_loss": update_reconstruction_loss,
            "reward_loss": update_reward_loss.item(),
            "sr_loss": update_sr_loss.item(),
            "norm_loss": update_norm_loss,
            "entropy": update_entropy,
            "value_loss": update_value_loss,
            "policy_loss": update_policy_loss,
            "grad_norm": update_grad_norm
        }

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
        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
