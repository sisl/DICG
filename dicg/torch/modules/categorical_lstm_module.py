"""CategoricalLSTMModule."""

import torch
from torch import nn
from torch.distributions import Categorical

class CategoricalLSTMModule(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size):

        super().__init__()

        # LSTM Args:
        # input_size: The number of expected features in the input `x`
        # hidden_size: The number of features in the hidden state `h`
        # num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
        #     would mean stacking two LSTMs together to form a `stacked LSTM`,
        #     with the second LSTM taking in outputs of the first LSTM and
        #     computing the final results. Default: 1
        # bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
        #     Default: ``True``
        # batch_first: If ``True``, then the input and output tensors are provided
        #     as (batch, seq, feature). Default: ``False``
        # dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
        #     LSTM layer except the last layer, with dropout probability equal to
        #     :attr:`dropout`. Default: 0
        # bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``


        # Inputs: input, (h_0, c_0)
        # - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
        #   of the input sequence.
        #   The input can also be a packed variable length sequence.
        #   See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
        #   :func:`torch.nn.utils.rnn.pack_sequence` for details.
        # - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
        #   containing the initial hidden state for each element in the batch.
        #   If the LSTM is bidirectional, num_directions should be 2, else it should be 1.
        # - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
        #   containing the initial cell state for each element in the batch.

        #   If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.


        # Outputs: output, (h_n, c_n)
        # - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
        #   containing the output features `(h_t)` from the last layer of the LSTM,
        #   for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
        #   given as the input, the output will also be a packed sequence.

        #   For the unpacked case, the directions can be separated
        #   using ``output.view(seq_len, batch, num_directions, hidden_size)``,
        #   with forward and backward being direction `0` and `1` respectively.
        #   Similarly, the directions can be separated in the packed case.
        # - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
        #   containing the hidden state for `t = seq_len`.

        #   Like *output*, the layers can be separated using
        #   ``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.
        # - **c_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
        #   containing the cell state for `t = seq_len`.

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size)
        self.decoder = nn.Linear(in_features=hidden_size,
                                 out_features=output_size)

    def forward(self, inputs, prev_hidden_state=None, prev_cell_state=None):
        if prev_cell_state is None:
            outputs, (next_hidden_state, next_cell_state) = self.lstm(inputs)
        else:
            outputs, (next_hidden_state, next_cell_state) = self.lstm(inputs, 
                (prev_hidden_state, prev_cell_state))
        outputs = self.decoder(outputs)
        dist = Categorical(logits=outputs)
        return dist, next_hidden_state, next_cell_state

        

