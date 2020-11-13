import akro
import torch
from torch import nn
import numpy as np

from torch.distributions import Categorical
from dicg.torch.modules import CategoricalMLPModule, DICGBase

class DICGCECategoricalMLPPolicy(DICGBase):
    def __init__(self,
                 env_spec,
                 n_agents,
                 encoder_hidden_sizes=(128, ),
                 embedding_dim=64,
                 attention_type='general',
                 n_gcn_layers=2,
                 residual=True,
                 gcn_bias=True,
                 categorical_mlp_hidden_sizes=(128, 64, 32),
                 name='dicg_ce_categorical_mlp_policy'):

        assert isinstance(env_spec.action_space, akro.Discrete), (
            'Categorical policy only works with akro.Discrete action space.')

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
        
        # Policy layer
        self.categorical_output_layer = \
            CategoricalMLPModule(input_dim=self._embedding_dim,
                                 output_dim=self._action_dim,
                                 hidden_sizes=categorical_mlp_hidden_sizes)
        self.layers.append(self.categorical_output_layer)

    def forward(self, obs_n, avail_actions_n):

        obs_n = torch.Tensor(obs_n)
        obs_n = obs_n.reshape(obs_n.shape[:-1] + (self._n_agents, -1))
        avail_actions_n = avail_actions_n.reshape(
            avail_actions_n.shape[:-1] + (self._n_agents, -1))

        embeddings_collection, attention_weights = super().forward(obs_n)

        # (n_paths, max_path_length, n_agents, action_space_dim)
        # or (n_agents, action_space_dim)
        if not self.residual:
            dists_n = \
                self.categorical_output_layer.forward(embeddings_collection[-1])
        else:
            embeddings_add = embeddings_collection[0] + embeddings_collection[-1]
            dists_n = self.categorical_output_layer.forward(embeddings_add)

        # Apply available actions mask
        masked_probs = dists_n.probs * torch.Tensor(avail_actions_n) # mask
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True) # renormalize
        masked_dists_n = Categorical(probs=masked_probs) # redefine distribution

        # masked_probs = dists_n.probs.clone()
        # masked_probs *= torch.Tensor(avail_actions_n) # mask
        # masked_probs /= masked_probs.sum(axis=-1, keepdims=True) # renormalize
        # masked_dists_n = Categorical(probs=masked_probs) # redefine distribution

        return masked_dists_n, attention_weights

    def get_actions(self, obs_n, avail_actions_n, greedy=False):
        """Independent agent actions (not using an exponential joint action space)
            
        Args:
            obs_n: list of obs of all agents in ONE time step [o1, o2, ..., on]
            E.g. 3 agents: [o1, o2, o3]

        """
        with torch.no_grad():
            dists_n, attention_weights = self.forward(obs_n, avail_actions_n)
            if not greedy:
                actions_n = dists_n.sample().numpy()
            else:
                actions_n = np.argmax(dists_n.probs.numpy(), axis=-1)
            agent_infos_n = {}
            agent_infos_n['action_probs'] = [dists_n.probs[i].numpy() 
                for i in range(len(actions_n))]
            agent_infos_n['attention_weights'] = [attention_weights.numpy()[i, :]
                for i in range(len(actions_n))]

            return actions_n, agent_infos_n

    def entropy(self, observations, avail_actions):
        # print('obs.shape =', observations.shape)
        dists_n, _ = self.forward(observations, avail_actions)
        # print('dist =', dists_n)
        # print('dist.probs =', dists_n.probs)
        entropy = dists_n.entropy()
        # print('entropy.shapeBefore =', entropy.shape)
        entropy = entropy.mean(axis=-1) # Asuming independent actions
        # print('entropy.shapeAfter =', entropy.shape)
        return entropy

    def log_likelihood(self, observations, avail_actions, actions):
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
        return False