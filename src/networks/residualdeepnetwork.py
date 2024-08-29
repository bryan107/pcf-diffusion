from typing import Tuple

import torch
import torch.nn as nn

from src.networks.residualnetwork import ResidualNetwork


class ResidualDeepNetwork(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: Tuple[int],
            flatten: bool = False,
    ):
        """
        Feedforward neural network with residual connection.
        Args:
            input_dim: integer, specifies input dimension of the neural network
            output_dim: integer, specifies output dimension of the neural network
            hidden_dims: list of integers, specifies the hidden dimensions of each layer.
                in above definition L = len(hidden_dims) since the last hidden layer is followed by an output layer
        """
        super().__init__()
        blocks = list()
        self.input_dim = input_dim
        self.flatten = flatten
        input_dim_block = input_dim

        tanh = torch.nn.Tanh()

        for hidden_dim in hidden_dims:
            blocks.append(ResidualNetwork(input_dim_block, hidden_dim))
            input_dim_block = hidden_dim
        blocks.append(tanh)
        blocks.append(nn.Linear(input_dim_block, output_dim))
        self.network = nn.Sequential(*blocks)
        self.blocks = blocks

    def forward(self, x):
        if self.flatten:
            x = x.reshape(x.shape[0], -1)
        out = self.network(x)
        return out
