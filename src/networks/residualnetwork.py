import torch.nn as nn


class ResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        # assert (
        #     input_dim == output_dim
        # ), "Input and output dimensions must be the same for a residual connection."
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.LeakyReLU()

        # TODO 07/08/2024 nie_k:  Hack, given that how this class is used might not be useful, no point in fixing it.
        self.create_residual_connection = True if input_dim == output_dim else False

    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.create_residual_connection:
            y = x + y
        return y
