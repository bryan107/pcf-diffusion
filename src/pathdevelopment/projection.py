from functools import partial

import numpy as np
import torch
import torch.nn as nn

from src.networks.nn import rescale_exp_matrix
from src.pathdevelopment.unitaryliealgebra import UnitaryLieAlgebra


class Projection(nn.Module):
    def __init__(self, input_size, hidden_size, channels=1, init_range=1, **kwargs):
        """
        Projection module used to project the path increments to the Lie group path increments
        using trainable weights from the Lie algebra.

        Args:
            input_size (int): Input size.
            hidden_size (int): Size of the hidden Lie algebra matrix.
            channels (int, optional): Number of channels to produce independent Lie algebra weights. Defaults to 1.
            init_range (int, optional): Range for weight initialization. Defaults to 1.
        """
        super().__init__()

        # Very cryptic way of adding parameters
        # self.__dict__.update(kwargs)

        self.channels = channels
        self.param_map = UnitaryLieAlgebra(hidden_size)
        self.measure_matrices = nn.Parameter(
            torch.empty(
                input_size, channels, hidden_size, hidden_size, dtype=torch.cfloat
            )
        )

        self.init_range = init_range
        self.set_matrices_projection()

        self.hidden_size = hidden_size

    def set_matrices_projection(self):
        UnitaryLieAlgebra.unitary_lie_init_(
            self.measure_matrices, partial(nn.init.normal_, std=1)
        )

    def M_initialize(self, A):
        init_range = np.linspace(0, 10, self.channels + 1)
        for i in range(self.channels):
            A[:, i] = UnitaryLieAlgebra.unitary_lie_init_(
                A[:, i], partial(nn.init.uniform_, a=init_range[i], b=init_range[i + 1])
            )
        return A

    def forward(self, dX: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projection module.

        Args:
            dX (torch.Tensor): Tensor of shape (N, input_size).

        Returns:
            torch.Tensor: Tensor of shape (N, channels, hidden_size, hidden_size).
        """
        A = self.param_map(self.measure_matrices).permute(1, 2, -1, 0)  # C,m,m,in
        A_dot_dX = A.matmul(dX.T).permute(-1, 0, 1, 2)  # ->C,m,m,N->N,C,m,m

        return rescale_exp_matrix(torch.linalg.matrix_exp, A_dot_dX)
