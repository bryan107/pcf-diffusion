import torch
import torch.nn as nn

from src.pathdevelopment.projection import Projection


class UnitaryDevelopmentLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        channels: int = 1,
        include_inital: bool = False,
    ):
        """
        Development layer module used for computation of unitary feature on time series.

        Args:
            input_size (int): Input size.
            hidden_size (int): Size of the hidden matrix.
            channels (int, optional): Number of channels. Defaults to 1.
            include_inital (bool, optional): Whether to include the initial value in the input. Defaults to False.
            time_batch (int, optional): Truncation value for batch processing. Defaults to 1.
        """
        super().__init__()
        self.input_size = input_size
        self.channels = channels
        self.hidden_size = hidden_size
        self.projection = Projection(input_size, hidden_size, channels)
        self.include_inital = include_inital
        return

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the development layer module.

        Args:
            input (torch.Tensor): Tensor with shape (N, L, input_size).

        Returns:
            torch.Tensor: Tensor with shape (N, L, hidden_size, hidden_size).
        """
        input = input.cfloat()

        N, L, D = input.shape
        if self.include_inital:
            input = torch.cat(
                [torch.zeros((N, 1, D), device=input.device), input], dim=1
            )

        # Consider the different sequence samples as different for the computation, and then reshaped back.
        diff_input_dx = (input[:, 1:] - input[:, :-1]).reshape(N * (L - 1), D)

        M_dX = self.projection(diff_input_dx).reshape(
            N, -1, self.channels, self.hidden_size, self.hidden_size
        )

        return self.dyadic_prod(M_dX)

    @staticmethod
    def dyadic_prod(X: torch.Tensor) -> torch.Tensor:
        """
        Computes the cumulative product on matrix time series with dyadic partitioning.

        Args:
            X (torch.Tensor): Batch of matrix time series of shape (N, T, C, m, m).

        Returns:
            torch.Tensor: Cumulative product on the time dimension of shape (N, T, m, m).
        """
        N, T, C, m, m = X.shape
        max_level = int(torch.ceil(torch.log2(torch.tensor(T))))
        I = (
            torch.eye(m, device=X.device, dtype=X.dtype)
            .reshape(1, 1, 1, m, m)
            .repeat(N, 1, C, 1, 1)
        )
        for i in range(max_level):
            if X.shape[1] % 2 == 1:
                X = torch.cat([X, I], 1)
            X = X.reshape(-1, 2, C, m, m)
            X = torch.einsum("bcij,bcjk->bcik", X[:, 0], X[:, 1])
            X = X.reshape(N, -1, C, m, m)

        return X[:, 0]
