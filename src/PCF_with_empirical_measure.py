import torch
from PIL import ImageFile
from torch import nn

from src.pathdevelopment.unitarydevelopmentlayer import UnitaryDevelopmentLayer

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PCF_with_empirical_measure(nn.Module):
    def __init__(
        self,
        num_samples,
        hidden_size,
        input_size,
        init_range: float = 1,
    ):
        """
        Class for computing the path charateristic function.

        Args:
            num_samples (int): Number of samples.
            hidden_size (int): Hidden size.
            input_size (int): Input size.
            add_time (bool): Whether to add time dimension to the input.
            init_range (float, optional): Range for weight initialization. Defaults to 1.
        """
        super().__init__()
        self.num_samples = num_samples
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.unitary_development = UnitaryDevelopmentLayer(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            channels=self.num_samples,
            include_inital=True,
            return_sequence=False,
            init_range=init_range,
        )
        for param in self.unitary_development.parameters():
            param.requires_grad = True

    @staticmethod
    def HS_norm(X: torch.tensor, Y: torch.Tensor):
        """
        Hilbert-Schmidt norm computation.

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

    def distance_measure(
        self, X1: torch.tensor, X2: torch.tensor, Lambda=0.1
    ) -> torch.float:
        """
        TODO: this description is just not true.
        Distance measure given by the Hilbert-Schmidt inner product.

        Args:
            X1 (torch.tensor): Time series samples with shape (N_1, T, d).
            X2 (torch.tensor): Time series samples with shape (N_2, T, d).
            Lambda (float, optional): Scaling factor for additional distance measure on the initial time point,
            this is found helpful for learning distribution of initial time point.
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

        mean_unitary_development_X_1 = self.unitary_development(X1).mean(0)
        mean_unitary_development_X_2 = self.unitary_development(X2).mean(0)
        diff_characteristic_function = (
            mean_unitary_development_X_1 - mean_unitary_development_X_2
        )

        if Lambda != 0:
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
            ) + Lambda * self.HS_norm(
                initial_CF_1 - initial_CF_2, initial_CF_1 - initial_CF_2
            )
        else:
            return self.HS_norm(
                diff_characteristic_function, diff_characteristic_function
            )
