import math

import torch
from torch import nn


class UnitaryLieAlgebra(nn.Module):
    """
    PyTorch module for parametrizing a unitary Lie algebra using a general linear matrix.

    The class handles anti-Hermitian matrices, which satisfy:

    .. math::
        A^\dagger = -A

    Here, :math:`A^\dagger` is the conjugate transpose of :math:`A`. This condition is necessary and sufficient for a matrix to belong to the Lie algebra of the unitary group :math:`U(n)`.

    - **Sufficiency:** The matrix exponential :math:`e^{At}` of any anti-Hermitian matrix :math:`A` is unitary.
    - **Necessity:** A matrix that belongs to the Lie algebra of the unitary group must satisfy :math:`A^\dagger = -A`.

    Thus, the set of anti-Hermitian matrices forms the Lie algebra of the unitary group.
    """

    @staticmethod
    def to_anti_hermitian(X: torch.Tensor) -> torch.Tensor:
        """
        Converts a matrix to its anti-Hermitian form.

        .. math::
            A = \\frac{X - X^\dagger}{2}

        Args:
            X (torch.Tensor): Input tensor of shape (..., n, n).

        Returns:
            torch.Tensor: Anti-Hermitian matrix of the same shape as input.
        """
        return (X - torch.conj(X.transpose(-2, -1))) / 2

    @staticmethod
    def in_lie_algebra(X: torch.Tensor, eps: float = 1e-5) -> bool:
        """
        Checks if the matrices defined by the last two axes belong to the unitary Lie algebra.

        Args:
            X (torch.Tensor): Tensor to check.
            eps (float): Tolerance for numerical comparison.

        Returns:
            bool: True if the tensor is in the Lie algebra, False otherwise.
        """
        return (
                X.dim() >= 2
                and X.size(-2) == X.size(-1)
                and torch.allclose(torch.conj(X.transpose(-2, -1)), -X, atol=eps)
        )

    @staticmethod
    def initialize_elements(
            tensor: torch.Tensor, distribution_fn: callable = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Initializes the diagonal and upper triangular elements of a tensor.

        Args:
            tensor (torch.Tensor): Multi-dimensional tensor where the last two dimensions form a square matrix.
            distribution_fn (callable, optional): Function to initialize the tensor with a specific distribution.
                Defaults to uniform distribution in the range :math:`[-\pi, \pi]`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Initialized diagonal, upper triangular real part,
            and upper triangular imaginary part.

        Raises:
            ValueError: If the tensor has fewer than 2 dimensions or the last two dimensions are not square.
        """
        if tensor.ndim < 2 or tensor.size(-1) != tensor.size(-2):
            raise ValueError(
                f"Expected a square matrix in the last two dimensions, got shape {tensor.size()}."
            )

        n: int = tensor.size(-2)
        size: int = tensor.size()[:-2]

        diag = tensor.new(size + (n,))
        off_diag = tensor.new(size + (2 * n, n))

        if distribution_fn is None:
            torch.nn.init.uniform_(diag, -math.pi, math.pi)
            torch.nn.init.uniform_(off_diag, -math.pi, math.pi)
        else:
            distribution_fn(diag)
            distribution_fn(off_diag)

        diag = diag.imag * 1j
        upper_tri_real = torch.triu(off_diag[..., :n, :n], 1).real.cfloat()
        upper_tri_complex = torch.triu(off_diag[..., n:, :n], 1).imag.cfloat() * 1j

        return diag, upper_tri_real, upper_tri_complex

    @staticmethod
    def unitary_lie_init_(
            tensor: torch.Tensor, distribution_fn: callable = None
    ) -> torch.Tensor:
        r"""
        Initializes a tensor in the unitary Lie algebra.

        Args:
            tensor (torch.Tensor): Tensor where the last two dimensions form a square matrix.
            distribution_fn (callable, optional): Function to initialize the tensor with a specific distribution.
                Defaults to uniform distribution in the range :math:`[-\pi, \pi]`.

        Returns:
            torch.Tensor: The initialized tensor.

        Raises:
            ValueError: If the tensor does not satisfy the unitary Lie algebra condition after initialization.
        """
        diag, upper_tri_real, upper_tri_complex = UnitaryLieAlgebra.initialize_elements(
            tensor, distribution_fn
        )

        real_part = (upper_tri_real - upper_tri_real.transpose(-2, -1)) / math.sqrt(2)
        complex_part = (
                               upper_tri_complex + upper_tri_complex.transpose(-2, -1)
                       ) / math.sqrt(2)

        with torch.no_grad():
            x = real_part + complex_part + torch.diag_embed(diag)
            if UnitaryLieAlgebra.in_lie_algebra(x):
                tensor.copy_(x.cfloat())
                return tensor
            raise ValueError(
                "Initialized tensor does not belong to the unitary Lie algebra."
            )

    def __init__(self, size: int):
        """
        Initialize the unitary module.

        Args:
            size (int): Size of the tensor to be parametrized.
        """
        super().__init__()
        self.size: int = size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Parametrizes the input tensor within the unitary Lie algebra.

        Args:
            X (torch.Tensor): Tensor of shape (..., 2n, 2n). The last two dimensions must form a square matrix.

        Returns:
            torch.Tensor: The transformed tensor within the unitary Lie algebra.

        Raises:
            ValueError: If the input tensor has fewer than 2 dimensions or if the last two dimensions are not square.
        """
        if X.ndim < 2 or X.size(-2) != X.size(-1):
            raise ValueError(
                "The input tensor must have at least 2 dimensions and the last two dimensions must form a square matrix."
            )
        return self.to_anti_hermitian(X)
