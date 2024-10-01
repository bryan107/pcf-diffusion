import torch
from PIL import ImageFile
from torch import nn

from src.pathdevelopment.unitarydevelopmentlayer import UnitaryDevelopmentLayer

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PCFEmpiricalMeasure(nn.Module):
    """
    A PyTorch module that computes a distance measure based on the Hilbert-Schmidt inner product
    between unitary developments of time series data.

    Args:
        num_samples (int): Number of samples in the time series.
        hidden_size (int): Hidden size for the unitary development layer.
        input_size (int): Input size for the time series data.
        init_range (float, optional): Initialization range for the unitary development layer. Defaults to 1.
        add_time (bool, optional): If True, adds a time dimension to the input data. Defaults to False.
    """

    @staticmethod
    def HS_norm(X: torch.Tensor, Y: torch.Tensor) -> torch.float:
        """
        Computes the Hilbert-Schmidt norm between two complex-valued matrices.

        The Hilbert-Schmidt norm is given by:

        .. math::
            ||X||_{HS} = \\sqrt{\\text{Tr}(X^H X)}

        where :math:`X^H` is the conjugate transpose of :math:`X`.

        Args:
            X (torch.Tensor): Complex-valued tensor of shape (C, m, m).
            Y (torch.Tensor): Tensor of the same shape as X.

        Returns:
            torch.float: Hilbert-Schmidt norm of X and Y.
        """
        assert (
            X.shape == Y.shape
        ), "X and Y must have the same shape but got {} and {}".format(X.shape, Y.shape)
        assert X.shape[-1] == X.shape[-2], "X must be square but got shape {}".format(
            X.shape
        )
        assert Y.shape[-1] == Y.shape[-2], "Y must be square but got shape {}".format(
            Y.shape
        )
        # TODO 11/08/2024 nie_k: actually these two asserts could be wrong, sometimes not the case.
        assert (
            X.dtype == torch.cfloat
        ), "X must be complex-valued but got dtype {}".format(X.dtype)
        assert (
            Y.dtype == torch.cfloat
        ), "Y must be complex-valued but got dtype {}".format(Y.dtype)

        if len(X.shape) == 4:
            m = X.shape[-1]
            X = X.reshape(-1, m, m)

        D = torch.bmm(X, torch.conj(Y).permute(0, 2, 1))
        return (torch.einsum("bii->b", D)).mean().real

    def __init__(
        self,
        num_samples: int,
        hidden_size: int,
        input_size: int,
        add_time: bool = False,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.add_time = add_time

        self.unitary_development = UnitaryDevelopmentLayer(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            channels=self.num_samples,
            include_inital=False,
        )

    def distance_measure(
        self, X1: torch.Tensor, X2: torch.Tensor, lambda_y: float = 0.1
    ) -> torch.float:
        """
        Computes a distance measure between two time series samples using the Hilbert-Schmidt norm.

        The distance measure is defined as:

        .. math::
            d(X_1, X_2) = ||\\varphi(X_1) - \\varphi(X_2)||_{HS} + \\lambda_y ||\\psi(X_1) - \\psi(X_2)||_{HS}

        where :math:`\\varphi` and :math:`\\psi` represent the unitary development
        for the full time series and the initial time point, respectively.

        Args:
            X1 (torch.Tensor): Time series samples with shape (N_1, T, d).
            X2 (torch.Tensor): Time series samples with shape (N_2, T, d).
            lambda_y (float, optional): Scaling factor for the distance measure on the initial time point.
                Defaults to 0.1.

        Returns:
            torch.float: Distance measure between two batches of samples.
        """
        N, T, d = X1.shape

        assert (
            X1.shape == X2.shape
        ), f"X1 and X2 must have the same shape but got {X1.shape} and {X2.shape}"
        assert (
            X1.shape[-1] == self.input_size
        ), f"X1 must have last dimension size {self.input_size} but got {X1.shape[-1]}"

        # Optionally add time to the data
        if self.add_time:
            X1 = add_time(X1)
            X2 = add_time(X2)

        # Compute the mean unitary developments
        mean_unitary_development_X_1 = self.unitary_development(X1).mean(0)
        mean_unitary_development_X_2 = self.unitary_development(X2).mean(0)
        diff_characteristic_function = (
            mean_unitary_development_X_1 - mean_unitary_development_X_2
        )

        if lambda_y > 1e-4:
            initial_incre_X1 = torch.cat(
                [torch.zeros((N, 1, d)).to(X1.device), X1[:, 0, :].unsqueeze(1)], dim=1
            )
            initial_incre_X2 = torch.cat(
                [torch.zeros((N, 1, d)).to(X1.device), X2[:, 0, :].unsqueeze(1)], dim=1
            )
            initial_CF_1 = self.unitary_development(initial_incre_X1).mean(0)
            initial_CF_2 = self.unitary_development(initial_incre_X2).mean(0)
            return self.HS_norm(
                diff_characteristic_function, diff_characteristic_function
            ) + lambda_y * self.HS_norm(
                initial_CF_1 - initial_CF_2, initial_CF_1 - initial_CF_2
            )
        else:
            return self.HS_norm(
                diff_characteristic_function, diff_characteristic_function
            )


def add_time(data: torch.Tensor) -> torch.Tensor:
    """
    Adds a time dimension to the input data.

    The time dimension is linearly spaced between 1/length and 1.

    Args:
        data (torch.Tensor): Input data of shape (N, T, d).

    Returns:
        torch.Tensor: Data with an added time dimension, shape (N, T, d+1).
    """
    size = data.shape[0]
    length = data.shape[1]
    tt = (
        torch.linspace(1 / length, 1, length)
        .reshape(1, -1, 1)
        .repeat(size, 1, 1)
        .to(data.device)
    )
    return torch.cat([tt, data], dim=-1)
