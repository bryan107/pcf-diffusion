from typing import Tuple

import torch
import torch.nn as nn

from src.networks.residualdeepnetwork import ResidualDeepNetwork
from src.utils.utils import init_weights


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
        self.use_bm_as_noise_else_gaussian = BM
        if BM:
            self.noise_scale = noise_scale
        else:
            self.noise_scale = 0.3

        return

    def get_noise_vector(self, shape: Tuple[int, ...], device: str) -> torch.Tensor:
        return self.noise_scale * torch.randn(*shape, self.input_dim, device=device)

    def forward(
        self,
        batch_size: int,
        n_lags: int,
        device: str,
        noise_sequence_z: torch.Tensor = None,
    ) -> torch.Tensor:
        noise_initial_hidden_states = self.get_noise_vector((batch_size,), device)

        # TODO 07/08/2024 nie_k: I have doubts regarding the usefullness of residual networks here.
        h0 = (
            self.initial_nn(noise_initial_hidden_states)
            .view(batch_size, self.rnn_num_layers, self.rnn_hidden_dim)
            .permute(1, 0, 2)
            .contiguous()
        )
        c0 = (
            self.initial_nn1(noise_initial_hidden_states)
            .view(batch_size, self.rnn_num_layers, self.rnn_hidden_dim)
            .permute(1, 0, 2)
            .contiguous()
        )

        if noise_sequence_z == None:
            noise_sequence_z = self.get_noise_vector((batch_size, n_lags), device)
            if self.use_bm_as_noise_else_gaussian:
                noise_sequence_z = noise_sequence_z.cumsum(1)

        hn, _ = self.rnn(noise_sequence_z, (h0, c0))
        output = self.linear(self.activation(hn))

        assert (
            output.shape[1] == n_lags
        ), f"output.shape[1] = {output.shape[1]} != {n_lags} = n_lags"

        return output
