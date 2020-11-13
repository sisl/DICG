"""CentralizedCategoricalMLPPolicy."""

import akro
import torch
from torch import nn
import numpy as np

from torch.distributions import Categorical
from garage.torch.modules import MLPModule

class CentralizedCategoricalMLPPolicy(MLPModule):
    def __init__(self,
                 env_spec,
                 n_agents,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='CentralizedCategoricalMLPPolicy'):
        assert isinstance(env_spec.action_space, akro.Discrete), (
            'Categorical policy only works with akro.Discrete action space.')

        self.centralized = True
        self.vectorized = True
        
        self._n_agents = n_agents
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.n

        self.name = name

        MLPModule.__init__(self,
                           input_dim=self._obs_dim,
                           output_dim=self._action_dim * self._n_agents,
                           hidden_sizes=hidden_sizes,
                           hidden_nonlinearity=hidden_nonlinearity,
                           hidden_w_init=hidden_w_init,
                           hidden_b_init=hidden_b_init,
                           output_nonlinearity=output_nonlinearity,
                           output_w_init=output_w_init,
                           output_b_init=output_b_init,
                           layer_normalization=layer_normalization)

    def grad_norm(self):
        return np.sqrt(
            np.sum([p.grad.norm(2).item() ** 2 for p in self.parameters()]))

    def forward(self, obs_n, avail_actions_n):
        """
        Idea: reshape last two dims of obs_n, feed forward and then reshape back
        Args:
            
            For get_actions(obs_n):
                obs_n.shape = (n_agents * obs_feat_dim, )
            For other purposes (e.g. likelihoods(), entropy()):
                obs_n.shape = (n_paths, max_path_length, n_agents * obs_feat_dim)
            
        """
        # Not reshapeing obs to make agents independent
        obs_n = torch.Tensor(obs_n)
        logits = super().forward(obs_n)
        # Make actions independent
        logits = logits.reshape(logits.shape[:-1] + (self._n_agents, -1))
        # Treating agents as being independent
        # Refer to Cooperative Multi-Agent Control Using Deep Reinforcement Learning
        dists_n = torch.distributions.Categorical(logits=logits)

        # Apply available actions mask
        avail_actions_n = avail_actions_n.reshape(
            avail_actions_n.shape[:-1] + (self._n_agents, -1))
        masked_probs = dists_n.probs * torch.Tensor(avail_actions_n) # mask
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True) # renormalize
        masked_dists_n = Categorical(probs=masked_probs) # redefine distribution
        return masked_dists_n

    def get_actions(self, obs_n, avail_actions_n, greedy=False):
        """Independent agent actions (not using an exponential joint action space)
            
        Args:
            obs_n: list of obs of all agents in ONE time step [o1, o2, ..., on]
            E.g. 3 agents: [o1, o2, o3]

        """
        with torch.no_grad():
            dists_n = self.forward(obs_n, avail_actions_n)
            actions_n = dists_n.sample().numpy()
            if not greedy:
                actions_n = dists_n.sample().numpy()
            else:
                actions_n = np.argmax(dists_n.probs.numpy(), axis=-1)
            agent_infos_n = {}
            agent_infos_n['action_probs'] = [dists_n.probs[i].numpy() 
                for i in range(len(actions_n))]
            return actions_n, agent_infos_n

    def reset(self, dones):
        pass

    def entropy(self, observations, avail_actions_n):
        dists_n = self.forward(observations, avail_actions_n)
        entropy = dists_n.entropy()
        entropy = entropy.mean(axis=-1) # Asuming independent actions
        return entropy

    def log_likelihood(self, observations, avail_actions_n, actions):
        dists_n = self.forward(observations, avail_actions_n)
        llhs = dists_n.log_prob(actions)
        # llhs.shape = (n_paths, max_path_length, n_agents)
        # For n agents action probability can be treated as independent
        # Pa = prob_i^n Pa_i
        # log(Pa) = sum_i^n log(Pa_i)
        llhs = llhs.sum(axis=-1) # Asuming independent actions
        # llhs.shape = (n_paths, max_path_length)
        return llhs

    @property
    def recurrent(self):
        return False

