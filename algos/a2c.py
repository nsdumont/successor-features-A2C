import numpy
import torch
import torch.nn.functional as F

from algos.base import BaseAlgo

class A2CAlgo(BaseAlgo):
    """The Advantage Actor-Critic algorithm."""

    def __init__(self, envs, model, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, entropy_decay=0,value_loss_coef=0.5, dissim_coef=0, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, model, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,entropy_decay,
                         value_loss_coef,dissim_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self, exps):
        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        # Initialize memory

        if self.model.recurrent:
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Create a sub-batch of experience

            sb = exps[inds + i]

            # Compute loss

            if self.model.recurrent:
                dist, value, memory = self.model(sb.obs, memory * sb.mask)
            else:
                dist, value, _ = self.model(sb.obs, None)

            if self.continuous_action:
                log_prob = dist.log_prob(sb.action)
                log_prob = torch.nan_to_num(log_prob, nan=0, posinf=10, neginf=-10)
                log_prob= log_prob.sum(dim=-1)
                entropy = -log_prob.mean()
                policy_loss = -(log_prob * sb.advantage).mean() 
                assert not torch.isnan(entropy).any()
                assert not torch.isnan(policy_loss).any()
            else:
                entropy = dist.entropy().mean()
                policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean() 
                
            
            value_loss = (value - sb.returnn).pow(2).mean()

            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
            assert not torch.isnan(loss).any()
            # if policy_loss < -1e5:
            #     print(policy_loss)
            # Update batch values

            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_loss += loss

        # Update update values

        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 if p.requires_grad else 0 for p in self.model.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values
        self.entropy_coef = self.entropy_coef*(1-self.entropy_decay)
        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
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
