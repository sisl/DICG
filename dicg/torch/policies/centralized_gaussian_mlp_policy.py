"""CentralizedCategoricalMLPPolicy."""

import akro
import torch
from torch import nn
import numpy as np

from torch.distributions import Normal, MultivariateNormal, Independent
from dicg.torch.modules import GaussianMLPModule

class CentralizedGaussianMLPPolicy(GaussianMLPModule):
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
                 share_std=False,
                 name='CentralizedGaussianMLPPolicy'):
        assert isinstance(env_spec.action_space, akro.Box), (
            'Gaussian policy only works with akro.Box action space.')

        self.centralized = True
        self.vectorized = True
        
        self._n_agents = n_agents
        self._obs_dim = env_spec.observation_space.flat_dim
        self._single_agent_action_dim = env_spec.action_space.shape[0]

        self.name = name
        self.share_std = share_std

        GaussianMLPModule.__init__(self,
            input_dim=self._obs_dim,
            output_dim=self._single_agent_action_dim * self._n_agents,
            single_agent_action_dim=self._single_agent_action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=True,
            # duplicate_std_copies=self._n_agents,
            share_std=share_std,
            init_std=1.0,
            min_std=1e-6,
            max_std=None,
            std_parameterization='exp',
            layer_normalization=False)

    def grad_norm(self):
        return np.sqrt(
            np.sum([p.grad.norm(2).item() ** 2 for p in self.parameters()]))

    def forward(self, obs_n, avail_actions_n=None):
        """
        Idea: reshape last two dims of obs_n, feed forward and then reshape back
        Args:
            
            For get_actions(obs_n):
                obs_n.shape = (n_agents * obs_feat_dim, )
            For other purposes (e.g. likelihoods(), entropy()):
                obs_n.shape = (n_paths, max_path_length, n_agents * obs_feat_dim)
            
        """
        # Not reshapeing obs to make it independent
        obs_n = torch.Tensor(obs_n)
        mean, std = super().forward(obs_n)
        # Make actions independent
        mean = mean.reshape(mean.shape[:-1] + (self._n_agents, -1))
        if self.share_std:
            std = std.reshape(std.shape[:-1] + (self._n_agents, -1))
            dist = Independent(Normal(mean, std), 1)
        else:
            # std = torch.diag(std.diagonal()[:self._single_agent_action_dim])
            dist = MultivariateNormal(mean, std)

        return dist

    def get_actions(self, obs_n, avail_actions_n=None, greedy=False):
        """Independent agent actions (not using an exponential joint action space)
            
        Args:
            obs_n: list of obs of all agents in ONE time step [o1, o2, ..., on]
            E.g. 3 agents: [o1, o2, o3]

        """
        with torch.no_grad():
            dists_n = self.forward(obs_n)
            if not greedy:
                actions_n = dists_n.sample().numpy()
            else:
                actions_n = dists_n.mean.numpy()
            agent_infos_n = {}
            agent_infos_n['action_mean'] = [dists_n.mean[i].numpy() 
                for i in range(len(actions_n))]
            agent_infos_n['action_std'] = [dists_n.stddev[i].numpy() 
                for i in range(len(actions_n))]

            return actions_n, agent_infos_n

    def reset(self, dones):
        pass

    def entropy(self, observations, avail_actions_n=None):
        dists_n = self.forward(observations)
        entropy = dists_n.entropy()
        entropy = entropy.mean(axis=-1) # Asuming independent actions
        return entropy

    def log_likelihood(self, observations, avail_actions_n, actions):
        dists_n = self.forward(observations)
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

