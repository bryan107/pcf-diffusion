from typing import Tuple

import torch
import torch.nn as nn

from src.networks.residualdeepnetwork import ResidualDeepNetwork
from src.utils.utils import init_weights


######################################################################################################
# Variant from lstmgenerator.py because we want to be able to sample in closed form. Common interface.
######################################################################################################


# TODO 12/08/2024 nie_k: appropriate name? Merge with decodedlstm?
#  It is the same thing but in the other one, the hidden states are not specified and x is the input noise.


### Task: rewrite this class with a free lstm. I have the class for it: Lstm_with_access_h0.
class LSTMGenerator_Diffusion(nn.Module):
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
            seq_len: int,
            dim_seq: int,
            num_diffusion_steps: int,
            device: str,
            noise_start_seq_z: torch.Tensor = None,
            alphas: torch.Tensor = None,
            betas: torch.Tensor = None,
            baralphas: torch.Tensor = None,
    ) -> torch.Tensor:
        # noise_seqs_z is used as input to rnn. If not specified, we take a random noise.
        # Should be before cumsum if you set the parameter.
        # noise_seqs_z is of shape (N, L, D).

        assert noise_start_seq_z.shape == (
            batch_size,
            seq_len,
            dim_seq,
        ), f"Expected shape (N, L, D), but got {noise_start_seq_z.shape}"

        assert (
                dim_seq == self.input_dim
        ), f"Expected input dim_seq with value {dim_seq} to be equal to {self.input_dim}"

        noise_initial_hidden_states = self.get_noise_vector(
            (batch_size, dim_seq), device
        )

        # TODO 07/08/2024 nie_k: I have doubts regarding the usefullness of residual networks here.
        hn = (
            self.initial_nn(noise_initial_hidden_states)
            .view(batch_size, self.rnn_num_layers, self.rnn_hidden_dim)
            # TODO 12/08/2024 nie_k: why permute
            .permute(1, 0, 2)
            .contiguous()
        )
        cn = (
            self.initial_nn1(noise_initial_hidden_states)
            .view(batch_size, self.rnn_num_layers, self.rnn_hidden_dim)
            # TODO 12/08/2024 nie_k: why permute
            .permute(1, 0, 2)
            .contiguous()
        )

        if noise_start_seq_z == None:
            noise_start_seq_z = self.get_noise_vector(
                (batch_size, seq_len, dim_seq), device
            )

        # Slicing to remove the time dimension of the input, which we do not need because it is deterministic.
        noise_start_seq_z = noise_start_seq_z[:, :, :-1]
        # WIP: I believe this is useless
        # if self.apply_cumsum_on_noise:
        #    noise_start_seq_z = noise_start_seq_z.cumsum(1)
        # WIP

        # Start of for loop.
        outputs = torch.zeros(
            num_diffusion_steps, batch_size, seq_len, self.output_dim, device=device
        )
        outputs[0] = noise_start_seq_z
        for i in range(1, seq_len):
            # Flatten L and D dimensions into one
            outputs_hidden, (hn, cn) = self.rnn(
                outputs[i - 1].flatten(1, 2).unsqueeze(1), (hn, cn)
            )
            # Use proper NN, the class already exists it is called basic_nn
            decoded = self.linear(self.activation(outputs_hidden))

            # todo: verify that the index for alphas are correct. I m confused atm.
            decoded = (
                    1.0
                    / torch.pow(alphas[seq_len - i], 0.5)
                    * (
                            outputs[i - 1]
                            - (1.0 - alphas[seq_len - i])
                            / torch.pow(1.0 - baralphas[seq_len - i], 0.5)
                            * decoded
                    )
            )
            if i + 1 != seq_len:
                decoded = decoded + torch.pow(betas[seq_len - i], 0.5) * torch.randn(
                    batch_size, seq_len, self.output_dim, device=device
                )
            outputs[i] = decoded
        # TODO 12/08/2024 nie_k: verify how to do it without flip.
        return outputs.flip(0)
