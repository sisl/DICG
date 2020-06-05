import akro
import torch
from torch import nn
import numpy as np
import copy

from torch.distributions import Normal
from dicg.torch.modules.gaussian_mlp_module import GaussianMLPModule
from dicg.torch.modules.dicg_base import DICGBase

class DICGCritic(DICGBase):

    def __init__(self,
                 env_spec,
                 n_agents,
                 encoder_hidden_sizes=(128, ),
                 embedding_dim=64,
                 decoder_hidden_sizes=(64, ),
                 attention_type='general',
                 n_gcn_layers=2,
                 residual=True,
                 gcn_bias=True,
                 share_std=False,
                 state_include_actions=False,
                 aggregator_type='sum',
                 name='dicg_critic'):

        super().__init__(
            env_spec=env_spec,
            n_agents=n_agents,
            encoder_hidden_sizes=encoder_hidden_sizes,
            embedding_dim=embedding_dim,
            attention_type=attention_type,
            n_gcn_layers=n_gcn_layers,
            gcn_bias=gcn_bias,
            state_include_actions=state_include_actions,
            name=name,
        )

        self.aggregator_type = aggregator_type
        if aggregator_type == 'sum':
            aggregator_input_dim = embedding_dim
        elif aggregator_type == 'direct':
            aggregator_input_dim = embedding_dim * self._n_agents

        self.baseline_aggregator = GaussianMLPModule(
            input_dim=aggregator_input_dim,
            output_dim=1,
            hidden_sizes=decoder_hidden_sizes,
            hidden_nonlinearity=torch.tanh,
            share_std=True,
        )

        self.residual = residual

    def compute_loss(self, obs_n, returns):
        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        embeddings_collection, attention_weights = super().forward(obs_n)
        if self.residual:
            emb = embeddings_collection[0] + embeddings_collection[-1]
        else:
            emb = embeddings_collection[-1]
        # shared std, any std = std.mean()
        if self.aggregator_type == 'sum':
            mean, std = self.baseline_aggregator(emb)     
            baseline_dist = Normal(mean.squeeze(-1).sum(-1), std.mean())
        elif self.aggregator_type == 'direct':
            emb = emb.reshape(emb.shape[:-2] + (-1, )) # concatenate embeddings
            mean, std = self.baseline_aggregator(emb)
            baseline_dist = Normal(mean.squeeze(-1), std.mean())
        ll = baseline_dist.log_prob(returns)
        return -ll.mean() # baseline loss

    def forward(self, obs_n):
        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        embeddings_collection, attention_weights = super().forward(obs_n)
        if self.residual:
            emb = embeddings_collection[0] + embeddings_collection[-1]
        else:
            emb = embeddings_collection[-1]

        if self.aggregator_type == 'sum':
            mean, _ = self.baseline_aggregator(emb)
            return mean.squeeze(-1).sum(-1)
        elif self.aggregator_type == 'direct':
            emb = emb.reshape(emb.shape[:-2] + (-1, ))
            mean, _ = self.baseline_aggregator(emb)
            return mean.squeeze(-1)

    def get_attention_weights(self, obs_n):
        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        _, attention_weights = super().forward(obs_n)
        return attention_weights