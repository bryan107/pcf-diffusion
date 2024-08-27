import logging
import sys
from abc import ABCMeta

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Lstm_with_access_h0(nn.Module, metaclass=ABCMeta):

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, (nn.GRU, nn.LSTM, nn.RNN)):
            for name, param in layer.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)
                continue

    def __init__(
        self,
        input_dim=1,
        num_layers=1,
        bidirectional: bool = False,
        nb_output_consider=1,
        hidden_size=150,
        dropout=0.0,
    ):
        super().__init_()
        self.input_dim = input_dim

        self.nb_output_consider = nb_output_consider

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.rnn_class = nn.LSTM

        self.stacked_rnn = self.rnn_class(
            self.input_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.apply(self.init_weights)
        return

    def forward(self, seqs, h0):
        out, h0 = self.stacked_rnn(
            seqs, h0
        )  # shape of out is  N,L,Hidden_size * (int(bidirectional) + 1)

        if self.bidirectional:
            # The shape of `out` is (N,  L, hidden_size * nb_directions).
            # We extract nb_output_consider elements: h_n, h_{n-1}, ...
            # The second dimension is reversed for the other direction.
            out = torch.cat(
                (
                    out[:, -self.nb_output_consider :, : self.hidden_size],
                    out[:, : self.nb_output_consider, self.hidden_size :],
                ),
                1,
            )

        else:
            # `out` is of shape (batch size, nb_output_consider, hidden_size)
            out = out[:, -self.nb_output_consider :, : self.hidden_size]

        return out, h0

    @property
    def device(self):
        # device of the model:
        # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
        try:
            return next(self.parameters()).device
        except StopIteration:
            print(
                "\n From Savable Net: no next parameter to determine the device. Program exits with error code 1."
            )
            sys.exit(1)

    @property
    def output_len(self):
        # Length of the output.
        return (
            self.hidden_size * (int(self.bidirectional) + 1) * self.nb_output_consider
        )
