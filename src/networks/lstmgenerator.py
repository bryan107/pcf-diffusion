from typing import Tuple

import torch
import torch.nn as nn

from src.networks.residualdeepnetwork import ResidualDeepNetwork
from src.utils.utils import init_weights


# TODO 12/08/2024 nie_k: appropriate name? Merge with decodedlstm?
#  It is the same thing but in the other one, the hidden states are not specified and x is the input noise.
class LSTMGenerator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        noise_scale=0.1,
        BM=False,
        activation=nn.Tanh(),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_hidden_dim = hidden_dim
        self.rnn_num_layers = n_layers

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.rnn_hidden_dim,
            num_layers=self.rnn_num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(self.rnn_hidden_dim, output_dim, bias=False)

        self.initial_nn = nn.Sequential(
            ResidualDeepNetwork(
                input_dim,
                self.rnn_hidden_dim * self.rnn_num_layers,
                [self.rnn_hidden_dim, self.rnn_hidden_dim],
            ),
            nn.Tanh(),
        )
        self.initial_nn1 = nn.Sequential(
            ResidualDeepNetwork(
                input_dim,
                self.rnn_hidden_dim * self.rnn_num_layers,
                [self.rnn_hidden_dim, self.rnn_hidden_dim],
            ),
            nn.Tanh(),
        )

        self.apply(init_weights)
        self.activation = activation

        # Noise config
        self.apply_cumsum_on_noise = BM
        if BM:
            self.noise_scale = noise_scale
        else:
            self.noise_scale = 0.3

        return

    def get_noise_vector(self, shape: Tuple[int, ...], device: str) -> torch.Tensor:
        return self.noise_scale * torch.randn(*shape, device=device)

    # todo: it is so weird to me we pass device, remove it. Cahnge n_lags name as well, i dont like it.
    def forward(
        self,
        batch_size: int,
        n_lags: int,
        device: str,
        noise_start_seq_z: torch.Tensor = None,
    ) -> torch.Tensor:
        # noise_seqs_z is used as input to rnn. If not specified, we take a random noise.
        # Should be before cumsum if you set the parameter.

        noise_initial_hidden_states = self.get_noise_vector(
            (batch_size, self.input_dim), device
        )

        # TODO 07/08/2024 nie_k: I have doubts regarding the usefullness of residual networks here.
        h0 = (
            self.initial_nn(noise_initial_hidden_states)
            .view(batch_size, self.rnn_num_layers, self.rnn_hidden_dim)
            # TODO 12/08/2024 nie_k: why permute
            .permute(1, 0, 2)
            .contiguous()
        )
        c0 = (
            self.initial_nn1(noise_initial_hidden_states)
            .view(batch_size, self.rnn_num_layers, self.rnn_hidden_dim)
            # TODO 12/08/2024 nie_k: why permute
            .permute(1, 0, 2)
            .contiguous()
        )

        if noise_start_seq_z == None:
            noise_start_seq_z = self.get_noise_vector(
                (batch_size, n_lags, self.input_dim), device
            )
        if self.apply_cumsum_on_noise:
            noise_start_seq_z = noise_start_seq_z.cumsum(1)

        hn, _ = self.rnn(noise_start_seq_z, (h0, c0))
        output = self.linear(self.activation(hn))
        return output
