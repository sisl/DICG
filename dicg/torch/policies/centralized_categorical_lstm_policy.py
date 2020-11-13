import akro
import torch
from torch import nn
import numpy as np
import copy

from torch.distributions import Categorical
from dicg.torch.modules import CategoricalLSTMModule, MLPEncoderModule

class CentralizedCategoricalLSTMPolicy(nn.Module):
    def __init__(self,
                 env_spec,
                 n_agents,
                 encoder_hidden_sizes=(64, 64),
                 embedding_dim=64,
                 lstm_hidden_size=64,
                 state_include_actions=False,
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='CentralizedCategoricalLSTMPolicy'):

        assert isinstance(env_spec.action_space, akro.Discrete), (
            'Categorical policy only works with akro.Discrete action space.')

        super().__init__()

        self.centralized = True
        self.vectorized = True
        
        self._n_agents = n_agents
        self._obs_dim = env_spec.observation_space.flat_dim # centralized obs_dim
        self._action_dim = env_spec.action_space.n
        self._embedding_dim = embedding_dim

        self.name = name

        self.state_include_actions = state_include_actions

        self._prev_actions = None
        self._prev_hiddens = None
        self._prev_cells = None

        if state_include_actions:
            mlp_input_dim = self._obs_dim + self._action_dim * n_agents
        else:
            mlp_input_dim = self._obs_dim
        
        self.encoder = MLPEncoderModule(
            input_dim=mlp_input_dim,
            output_dim=self._embedding_dim,
            hidden_sizes=encoder_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

        self.categorical_lstm_output_layer = \
            CategoricalLSTMModule(input_size=self._embedding_dim,
                                  output_size=self._action_dim * n_agents,
                                  hidden_size=lstm_hidden_size)

    def grad_norm(self):
        return np.sqrt(
            np.sum([p.grad.norm(2).item() ** 2 for p in self.parameters()]))

    # Batch forward
    def forward(self, obs_n, avail_actions_n, actions_n=None):

        obs_n = torch.Tensor(obs_n)
        n_paths = obs_n.shape[0]
        max_path_len = obs_n.shape[1]
        if self.state_include_actions:
            obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
            assert actions_n is not None
            actions_n = torch.Tensor(actions_n).unsqueeze(-1).type(torch.LongTensor)
            # actions_n.shape = (n_paths, max_path_len, n_agents, 1)
            # Convert actions_n to one hot encoding
            actions_onehot = torch.zeros(actions_n.shape[:-1] + (self._action_dim,))
            # actions_onehot.shape = (n_paths, max_path_len, n_agents, action_dim)
            actions_onehot.scatter_(-1, actions_n, 1)
            # Shift and pad actions_onehot by one time step
            actions_onehot_shifted = actions_onehot[:, :-1, :, :]
            # Use zeros as _prev_actions in the first time step
            zero_pad = torch.zeros(n_paths, 1, self._n_agents, self._action_dim)
            # Concatenate zeros to the beginning of actions
            actions_onehot_shifted = torch.cat((zero_pad, actions_onehot_shifted), dim=1)
            # Combine actions into obs
            obs_n = torch.cat((obs_n, actions_onehot_shifted), dim=-1)
            # Reshape obs_n back to concatenated centralized form
            obs_n = obs_n.reshape(n_paths, max_path_len, -1) # One giant vector

        inputs = self.encoder.forward(obs_n)

        # inputs.shape = (n_paths, max_path_len, n_agents * emb_dim) 
        # Reshape to be compliant with the input shape requirement of LSTM
        inputs = inputs.transpose(0, 1)
        # inputs.shape = (max_path_len, n_paths, n_agents * emb_dim)

        dists_n = self.categorical_lstm_output_layer.forward(inputs)[0]

        # Apply available actions mask
        avail_actions_n = avail_actions_n.reshape(
            avail_actions_n.shape[:-1] + (self._n_agents, -1))
        # Reshape back
        masked_probs = dists_n.probs.reshape(
            max_path_len, n_paths, self._n_agents, self._action_dim)
        masked_probs = masked_probs.transpose(0, 1)
        masked_probs = masked_probs * torch.Tensor(avail_actions_n) # mask
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True) # renormalize
        masked_dists_n = Categorical(probs=masked_probs) # redefine distribution

        return masked_dists_n

    def step_forward(self, obs_n, avail_actions_n):
        """
            Single step forward for stepping in envs
        """
        # obs_n.shape = (n_envs, n_agents * obs_dim)
        obs_n = torch.Tensor(obs_n)
        n_envs = obs_n.shape[0]
        if self.state_include_actions:
            obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
            if self._prev_actions is None:
                self._prev_actions = torch.zeros(n_envs, self._n_agents, self._action_dim)
            obs_n = torch.cat((obs_n, self._prev_actions), dim=-1)
            # Reshape obs_n back to concatenated centralized form
            obs_n = obs_n.reshape(n_envs, -1) # One giant vector
            # obs_n.shape = (n_envs, n_agents * (obs_dim + action_dim))

        inputs = self.encoder.forward(obs_n)

        # input.shape = (n_envs, n_agents * emb_dim)
        inputs = inputs.reshape(1, n_envs, -1)

        dists_n, next_h, next_c = self.categorical_lstm_output_layer.forward(
                inputs, self._prev_hiddens, self._prev_cells)

        self._prev_hiddens = next_h
        self._prev_cells = next_c

        # Apply available actions mask
        avail_actions_n = avail_actions_n.reshape(
            avail_actions_n.shape[:-1] + (self._n_agents, -1))
        masked_probs = dists_n.probs.squeeze(0).reshape(
            n_envs, self._n_agents, self._action_dim)
        masked_probs = masked_probs * torch.Tensor(avail_actions_n) # mask
        masked_probs = masked_probs / masked_probs.sum(axis=-1, keepdims=True) # renormalize
        masked_dists_n = Categorical(probs=masked_probs) # redefine distribution

        return masked_dists_n



    def get_actions(self, obs_n, avail_actions_n, greedy=False):
        """Independent agent actions (not using an exponential joint action space)
            
        Args:
            obs_n: list of obs of all agents in ONE time step [o1, o2, ..., on]
            E.g. 3 agents: [o1, o2, o3]

        """
        with torch.no_grad():
            dists_n = self.step_forward(obs_n, avail_actions_n)
            if not greedy:
                actions_n = dists_n.sample().numpy()
            else:
                actions_n = np.argmax(dists_n.probs.numpy(), axis=-1)
            agent_infos_n = {}
            agent_infos_n['action_probs'] = [dists_n.probs[i].numpy() 
                for i in range(len(actions_n))]

            if self.state_include_actions:
                # actions_onehot.shape = (n_envs, self._n_agents, self._action_dim)
                actions_onehot = torch.zeros(len(obs_n), self._n_agents, self._action_dim)
                actions_onehot.scatter_(
                    -1, torch.Tensor(actions_n).unsqueeze(-1).type(torch.LongTensor), 1)     
                self._prev_actions = actions_onehot

            return actions_n, agent_infos_n

    def reset(self, dones):
        if all(dones): # dones is synched
            self._prev_actions = None
            self._prev_hiddens = None
            self._prev_cells = None

    def entropy(self, observations, avail_actions, actions=None):
        # print('obs.shape =', observations.shape)
        dists_n = self.forward(observations, avail_actions, actions)
        entropy = dists_n.entropy()
        entropy = entropy.mean(axis=-1) # Asuming independent actions
        return entropy

    def log_likelihood(self, observations, avail_actions, actions):
        if self.state_include_actions:
            dists_n = self.forward(observations, avail_actions, actions)
        else:
            dists_n = self.forward(observations, avail_actions)
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
    

