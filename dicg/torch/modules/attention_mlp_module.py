import akro
import torch
from torch import nn
import numpy as np

from torch.distributions import Categorical
from garage.torch.policies import Policy
# from garage.torch.modules.mlp_module import MLPModule

from .mlp_encoder_module import MLPEncoderModule
from .gaussian_mlp_module import GaussianMLPModule
from .attention_module import AttentionModule

class AttentionMLP(nn.Module):
    def __init__(self,
                 env_spec,
                 n_agents,
                 mode,
                 encoder_hidden_sizes=(128, ),
                 embedding_dim=64,
                 attention_type='general',
                 state_include_actions=False,
                 name='attention_mlp'):

        super().__init__()

        self.name = name

        self.centralized = True
        self.vectorized = True
        
        self._n_agents = n_agents
        self._cent_obs_dim = env_spec.observation_space.flat_dim
        self._dec_obs_dim = int(self._cent_obs_dim / n_agents)
        if isinstance(env_spec.action_space, akro.Discrete): 
            self._action_dim = env_spec.action_space.n
        else:
            self._action_dim = env_spec.action_space.shape[0]
        self._embedding_dim = embedding_dim

        if state_include_actions:
            self._dec_obs_dim += self._action_dim
        
        self.encoder = MLPEncoderModule(input_dim=self._dec_obs_dim,
                                        output_dim=self._embedding_dim,
                                        hidden_sizes=encoder_hidden_sizes,
                                        output_nonlinearity=torch.tanh)

        self.self_attention = AttentionModule(dimensions=self._embedding_dim, 
                                              attention_type=attention_type)
        self.mode = mode
        if self.mode == 'critic':
            self.post_processer = GaussianMLPModule(
                input_dim=(self._embedding_dim + self._n_agents) * self._n_agents,
                output_dim=1,
                hidden_sizes=(256, 64),
                hidden_nonlinearity=torch.tanh,
                share_std=True,
            )
        elif self.mode == 'obs_encoder':
            self.post_processer = MLPEncoderModule(
                input_dim=self._embedding_dim + self._n_agents,
                output_dim=self._embedding_dim,
                hidden_sizes=(64, ),
                output_nonlinearity=torch.tanh)


    def grad_norm(self):
        return np.sqrt(
            np.sum([p.grad.norm(2).item() ** 2 for p in self.parameters()]))

    def forward(self, obs_n):
        # (n_paths, max_path_length, n_agents, emb_feat_dim)
        # or (n_agents, emb_feat_dim)
        embeddings = self.encoder.forward(obs_n)

        # (n_paths, max_path_length, n_agents, n_agents)
        # or (n_agents, n_agents)
        attention_weights = self.self_attention.forward(embeddings)

        embeddings_concat = torch.cat((embeddings, attention_weights), dim=-1)
        if self.mode == 'obs_encoder':
            # last dim dec forward
            embeddings_processed = self.post_processer.forward(embeddings_concat)
            return embeddings_processed, attention_weights
        elif self.mode == 'critic':
            # last dim concatenate forward
            embeddings_concat = embeddings_concat.reshape(embeddings_concat.shape[:-2] + (-1, ))
            baseline_mean, baseline_std = self.post_processer.forward(embeddings_concat)
            return baseline_mean, baseline_std, attention_weights


    def reset(self, dones):
        pass

    

