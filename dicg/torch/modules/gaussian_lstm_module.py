"""GaussianLSTMModule."""

import torch
from torch import nn
from torch.distributions import Normal
        

class GaussianLSTMModule(nn.Module):
    """GaussianLSTMModule.

        Adapted from garage.torch.module.GaussianMLPModule
    
    Some Args are not used.
    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_dim (list[int]): Hidden state dimension of LSTM layer for the 
            mean.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network.
        std_hidden_w_init (callable):  Initializer function for the weight
            of hidden layer (s).
        std_hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s).
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        std_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the std network.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation.
            - softplus: the std will be computed as log(1+exp(x)).
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 single_agent_action_dim=None, # used for centralized
                 hidden_dim=64,
                 share_std=False,
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_nonlinearity=torch.tanh,
                 std_hidden_w_init=nn.init.xavier_uniform_,
                 std_hidden_b_init=nn.init.zeros_,
                 std_output_nonlinearity=None,
                 std_output_w_init=nn.init.xavier_uniform_,
                 std_parameterization='exp',
                 layer_normalization=False):
        super().__init__()

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._action_dim = output_dim
        self._single_agent_action_dim = single_agent_action_dim
        self._learn_std = learn_std
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._std_hidden_w_init = std_hidden_w_init
        self._std_hidden_b_init = std_hidden_b_init
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_output_w_init = std_output_w_init
        self._std_parameterization = std_parameterization
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization

        if self._std_parameterization not in ('exp', 'softplus'):
            raise NotImplementedError

        self._share_std = share_std
        if share_std:
            init_std_param = torch.Tensor([init_std]).log()
        else:
            if single_agent_action_dim is not None:
                init_std_param = torch.Tensor([init_std] * single_agent_action_dim).log()
            else:
                init_std_param = torch.Tensor([init_std] * self._action_dim).log()
        
        if self._learn_std:
            self._init_std = torch.nn.Parameter(init_std_param)
        else:
            self._init_std = init_std_param

        self._min_std_param = self._max_std_param = None
        if min_std is not None:
            self._min_std_param = torch.Tensor([min_std]).log()
        if max_std is not None:
            self._max_std_param = torch.Tensor([max_std]).log()

        self._mean_lstm = nn.LSTM(input_size=input_dim,
                                  hidden_size=hidden_dim)
        
        self.mean_decoder = nn.Linear(in_features=hidden_dim,
                                      out_features=output_dim)


    def forward(self, inputs, prev_hidden_state=None, prev_cell_state=None):
        """Forward method.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: Module output.

        """
        if prev_cell_state is None:
            mean_outputs, (next_hidden_state, next_cell_state) = \
                self._mean_lstm(inputs)
        else:
            mean_outputs, (next_hidden_state, next_cell_state) = \
                self._mean_lstm(inputs, (prev_hidden_state, prev_cell_state))

        mean = self.mean_decoder(mean_outputs)

        if self._share_std:
            broadcast_shape = list(inputs.shape[:-1]) + [self._action_dim]
            log_std_uncentered = torch.zeros(*broadcast_shape) + self._init_std
        else:
            log_std_uncentered = self._init_std

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=self._to_scalar_if_not_none(self._min_std_param),
                max=self._to_scalar_if_not_none(self._max_std_param))

        if self._share_std:
            if self._std_parameterization == 'exp':
                std = log_std_uncentered.exp()
            else:
                std = log_std_uncentered.exp().exp().add(1.).log()
        else:
            if self._std_parameterization == 'exp':
                std = torch.diag(log_std_uncentered.exp())
            else:
                std = torch.diag(log_std_uncentered.exp().exp().add(1.).log())

        # dist = Independent(Normal(mean, std), 1) # Independent?
        # dist = Normal(mean, std)

        return mean, std, next_hidden_state, next_cell_state

    # pylint: disable=no-self-use
    def _to_scalar_if_not_none(self, tensor):
        """Convert torch.Tensor of a single value to a Python number.

        Args:
            tensor (torch.Tensor): A torch.Tensor of a single value.

        Returns:
            float: The value of tensor.

        """
        return None if tensor is None else tensor.item()

        