import akro
import torch
from torch import nn
import numpy as np

from dicg.torch.modules import GaussianMLPModule, DICGBase
from torch.distributions import Normal, MultivariateNormal, Independent

class DICGCEGaussianMLPPolicy(DICGBase):
    def __init__(self,
                 env_spec,
                 n_agents,
                 encoder_hidden_sizes=(128,),
                 embedding_dim=64,
                 attention_type='general',
                 n_gcn_layers=2,
                 residual=True,
                 gcn_bias=True,
                 gaussian_mlp_hidden_sizes=(128, 64, 32),
                 share_std=False,
                 name='dicg_ce_gaussian_mlp_policy'):

        assert isinstance(env_spec.action_space, akro.Box), (
            'Gaussian policy only works with akro.Box action space.')

        super().__init__(
            env_spec=env_spec,
            n_agents=n_agents,
            encoder_hidden_sizes=encoder_hidden_sizes,
            embedding_dim=embedding_dim,
            attention_type=attention_type,
            n_gcn_layers=n_gcn_layers,
            gcn_bias=gcn_bias,
            name=name
        )
        self.residual = residual
        self.share_std = share_std
        
        # Policy layer
        self.gaussian_output_layer = GaussianMLPModule(
            input_dim=self._embedding_dim,
            output_dim=self._action_dim,
            hidden_sizes=gaussian_mlp_hidden_sizes,
            share_std=share_std)
        self.layers.append(self.gaussian_output_layer)

    def forward(self, obs_n, avail_actions_n=None):

        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))

        embeddings_collection, attention_weights = super().forward(obs_n)

        # (n_paths, max_path_length, n_agents, action_space_dim)
        # or (n_agents, action_space_dim)
        if not self.residual:
            mean, std = \
                self.gaussian_output_layer.forward(embeddings_collection[-1])
            dists_n = Independent(Normal(mean, std), 1)
        else:
            embeddings_add = embeddings_collection[0] + embeddings_collection[-1]
            mean, std = self.gaussian_output_layer.forward(embeddings_add)
            dists_n = MultivariateNormal(mean, std)

        return dists_n, attention_weights

    def get_actions(self, obs_n, avail_actions_n=None, greedy=False):
        """Independent agent actions (not using an exponential joint action space)
            
        Args:
            obs_n: list of obs of all agents in ONE time step [o1, o2, ..., on]
            E.g. 3 agents: [o1, o2, o3]

        """
        with torch.no_grad():
            dists_n, attention_weights = self.forward(obs_n)
            if not greedy:
                actions_n = dists_n.sample().numpy()
            else:
                actions_n = dists_n.mean.numpy()
            agent_infos_n = {}
            agent_infos_n['action_mean'] = [dists_n.mean[i].numpy() 
                for i in range(len(actions_n))]
            agent_infos_n['action_std'] = [dists_n.stddev[i].numpy() 
                for i in range(len(actions_n))]
            agent_infos_n['attention_weights'] = [attention_weights.numpy()[i, :]
                for i in range(len(actions_n))]

            return actions_n, agent_infos_n

    def entropy(self, observations, avail_actions=None):
        dists_n, _ = self.forward(observations)
        entropy = dists_n.entropy()
        entropy = entropy.mean(axis=-1) # Asuming independent actions
        return entropy

    def log_likelihood(self, observations, avail_actions, actions):
        dists_n, _ = self.forward(observations)
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