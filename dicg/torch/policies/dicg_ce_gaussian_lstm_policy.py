import akro
import torch
from torch import nn
import numpy as np
import copy

from torch.distributions import Normal, MultivariateNormal, Independent
from dicg.torch.modules import GaussianLSTMModule, DICGBase

class DICGCEGaussianLSTMPolicy(DICGBase):
    def __init__(self,
                 env_spec,
                 n_agents,
                 encoder_hidden_sizes=(128, ),
                 embedding_dim=64,
                 attention_type='general',
                 n_gcn_layers=2,
                 residual=True,
                 gcn_bias=True,
                 lstm_hidden_size=64,
                 share_std=False,
                 state_include_actions=False,
                 name='dicg_ce_categorical_mlp_policy'):

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
            state_include_actions=state_include_actions,
            name=name
        )
        self.residual = residual
        self.state_include_actions = state_include_actions
        self.share_std = share_std

        self._prev_actions = None
        self._prev_hiddens = None
        self._prev_cells = None
        
        # Policy layer
        self.gaussian_lstm_output_layer = \
            GaussianLSTMModule(input_dim=self._embedding_dim,
                               output_dim=self._action_dim,
                               hidden_dim=lstm_hidden_size,
                               share_std=share_std)
        self.layers.append(self.gaussian_lstm_output_layer)

    # Batch forward
    def forward(self, obs_n, avail_actions_n=None, actions_n=None):
        """
            avail_actions_n is not used for continuous action space
        """

        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        n_paths = obs_n.shape[0]
        max_path_len = obs_n.shape[1]

        if self.state_include_actions:
            assert actions_n is not None
            # actions_n = torch.Tensor(actions_n)
            # actions_n.shape = (n_paths, max_path_len, n_agents, action_dim)
            # Shift and pad actions by one time step
            actions_shifted = actions_n[:, :-1, :, :]
            # Use zeros as _prev_actions in the first time step
            zero_pad = torch.zeros(n_paths, 1, self._n_agents, self._action_dim)
            # Concatenate zeros to the beginning of actions
            actions_shifted = torch.cat((zero_pad, actions_shifted), dim=1)
            # Combine actions into obs
            obs_n = torch.cat((obs_n, actions_shifted), dim=-1)

        embeddings_collection, attention_weights = super().forward(obs_n)

        if self.residual:
            inputs = embeddings_collection[0] + embeddings_collection[-1]
        else:
            inputs = embeddings_collection[-1]

        # inputs.shape = (n_paths, max_path_len, n_agents, emb_dim) 
        inputs = inputs.transpose(0, 1)
        # inputs.shape = (max_path_len, n_paths, n_agents, emb_dim)
        inputs = inputs.reshape(
            max_path_len, n_paths * self._n_agents, self._embedding_dim)

        mean, std, _, _ = self.gaussian_lstm_output_layer.forward(inputs)
        # mean.shape = (max_path_len, n_paths * n_agents, action_dim)
        # Need to reshape back dists_n
        mean = mean.reshape(max_path_len, n_paths, self._n_agents, 
                            self._action_dim).transpose(0, 1)
        if self.share_std:
            std = std.reshape(max_path_len, n_paths, self._n_agents, 
                            self._action_dim).transpose(0, 1)
            dists_n = Independent(Normal(mean, std), 1)
        else:
            dists_n = MultivariateNormal(mean, std)
            
        return dists_n, attention_weights

    def step_forward(self, obs_n, avail_actions_n=None):
        """
            Single step forward for stepping in envs
        """
        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        n_envs = obs_n.shape[0]

        if self.state_include_actions:
            if self._prev_actions is None:
                self._prev_actions = torch.zeros(n_envs, self._n_agents, self._action_dim)
            obs_n = torch.cat((obs_n, self._prev_actions), dim=-1)

        embeddings_collection, attention_weights = super().forward(obs_n)

        if self.residual:
            inputs = embeddings_collection[0] + embeddings_collection[-1]
        else:
            inputs = embeddings_collection[-1]

        # input.shape = (n_envs, n_agents, emb_dim)
        inputs = inputs.reshape(
            1, n_envs * self._n_agents, self._embedding_dim)

        mean, std, next_h, next_c = self.gaussian_lstm_output_layer.forward(
                inputs, self._prev_hiddens, self._prev_cells)

        self._prev_hiddens = next_h
        self._prev_cells = next_c

        # Need to reshape mean and std back
        mean = mean.reshape(n_envs, self._n_agents, self._action_dim)
        if self.share_std:
            std = std.reshape(n_envs, self._n_agents, self._action_dim)
            dists_n = Independent(Normal(mean, std), 1)
        else:
            dists_n = MultivariateNormal(mean, std)

        return dists_n, attention_weights



    def get_actions(self, obs_n, avail_actions_n=None, greedy=False):
        """Independent agent actions (not using an exponential joint action space)
            
        Args:
            obs_n: list of obs of all agents in ONE time step [o1, o2, ..., on]
            E.g. 3 agents: [o1, o2, o3]

        """
        with torch.no_grad():
            dists_n, attention_weights = self.step_forward(obs_n)
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

            if self.state_include_actions:
                # actions_n.shape = (n_envs, self._n_agents, self._action_dim)
                self._prev_actions = torch.Tensor(actions_n)

            return actions_n, agent_infos_n

    def reset(self, dones):
        if all(dones): # dones is synched
            self._prev_actions = None
            self._prev_hiddens = None
            self._prev_cells = None

    def entropy(self, observations, avail_actions=None, actions=None):
        dists_n, _ = self.forward(observations, avail_actions, actions)
        entropy = dists_n.entropy()
        entropy = entropy.mean(axis=-1) # Asuming independent actions
        return entropy

    def log_likelihood(self, observations, avail_actions, actions):
        avail_actions = None
        if self.state_include_actions:
            dists_n, _ = self.forward(observations, avail_actions, actions)
        else:
            dists_n, _ = self.forward(observations, avail_actions)
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
        return True
    

