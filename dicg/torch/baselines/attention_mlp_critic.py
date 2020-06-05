import akro
import torch
from torch import nn
import numpy as np
import copy

from torch.distributions import Normal
from dicg.torch.modules.gaussian_mlp_module import GaussianMLPModule
from dicg.torch.modules.attention_mlp_module import AttentionMLP

class AttentionMLPCritic(AttentionMLP):

    def __init__(self,
                 env_spec,
                 n_agents,
                 encoder_hidden_sizes=(128, ),
                 embedding_dim=64,
                 decoder_hidden_sizes=(64, ),
                 attention_type='general',
                 share_std=False,
                 state_include_actions=False,
                 name='attention_mlp_critic'):

        super().__init__(
            env_spec=env_spec,
            n_agents=n_agents,
            mode='critic',
            encoder_hidden_sizes=encoder_hidden_sizes,
            embedding_dim=embedding_dim,
            attention_type=attention_type,
            state_include_actions=state_include_actions,
            name=name,
        )

    def compute_loss(self, obs_n, returns):
        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        mean, std, attention_weights = super().forward(obs_n)
        # shared std, any std = std.mean()
        baseline_dist = Normal(mean.squeeze(-1), std.mean())
        ll = baseline_dist.log_prob(returns)
        return -ll.mean() # baseline loss

    def forward(self, obs_n):
        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        mean, std, attention_weights = super().forward(obs_n)
        return mean.squeeze(-1)

    def get_attention_weights(self, obs_n):
        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        _, _, attention_weights = super().forward(obs_n)
        return attention_weights