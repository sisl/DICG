import math

import torch
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np

class GraphConvolutionModule(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, id=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.id = id
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        if len(inputs.shape) == 2:
            support = torch.mm(inputs, self.weight)
            outputs = torch.mm(adj, support)
        elif len(inputs.shape) > 2:
            # n_paths, max_path_length, n_agents, emb_feat_dim = inputs.size()
            support = torch.matmul(inputs, self.weight)
            # adj dim = (n_paths, max_path_length, n_agents, n_agents)
            # support dim = (n_paths, max_path_length, n_agents, emb_feat_dim)
            outputs = torch.matmul(adj, support)
            # outputs dim = (n_paths, max_path_length, n_agents, emb_feat_dim)

        if self.bias is not None:
            return torch.tanh(outputs + self.bias)
        else:
            return torch.tanh(outputs)
