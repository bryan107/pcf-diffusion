"""
This module provides functions for working with anti-Hermitian matrices, which are key elements of the Lie algebra
of the unitary group U(n). The module includes tools for converting matrices to anti-Hermitian form, checking if
matrices belong to the unitary Lie algebra, and initializing matrices in this algebra.

The core mathematical properties and operations revolve around anti-Hermitian matrices, which satisfy:

.. math::
    A^\dagger = -A

Here, :math:`A^\dagger` denotes the conjugate transpose of :math:`A`. This property is both necessary and sufficient
for a matrix to belong to the Lie algebra of the unitary group U(n):

- **Sufficiency:** The matrix exponential :math:`e^{At}` of any anti-Hermitian matrix :math:`A` is unitary.
- **Necessity:** Any matrix that belongs to the Lie algebra of the unitary group must satisfy :math:`A^\dagger = -A`.

As a result, the set of anti-Hermitian matrices forms the Lie algebra of the unitary group.
"""

import math

import torch


def to_anti_hermitian(X: torch.Tensor, dim1: int = -2, dim2: int = -1) -> torch.Tensor:
    """
    Converts a given matrix to its anti-Hermitian form over specified dimensions.

    An anti-Hermitian matrix satisfies the property :math:`A^\dagger = -A`, where :math:`A^\dagger` is the conjugate
    transpose of :math:`A`. This function computes the anti-Hermitian matrix as:

    .. math::
        A = \\frac{X - X^\dagger}{2}

    Args:
        X (torch.Tensor): Input tensor.
        dim1 (int): The first dimension to be used in the conjugate transpose.
        dim2 (int): The second dimension to be used in the conjugate transpose.

    Returns:
        torch.Tensor: Anti-Hermitian matrix of the same shape as the input.

    Raises:
        ValueError: If the specified dimensions are not valid (e.g., out of range or the same).
    """
    assert (
        X.ndim >= 2
    ), f"The input tensor must have at least 2 dimensions, but got {X.ndim} dimensions. Tensor shape: {X.shape}"
    assert X.size(dim1) == X.size(
        dim2
    ), f"The last two dimensions of the tensor must form a square matrix, but got shapes {X.size(-2)} and {X.size(-1)}. Tensor shape: {X.shape}"
    assert dim1 != dim2, "The specified dimensions must be different."
    assert (
        dim1 < X.ndim
    ), f"Dimension {dim1} is out of range for a tensor with {X.ndim} dimensions."
    assert (
        dim2 < X.ndim
    ), f"Dimension {dim2} is out of range for a tensor with {X.ndim} dimensions."

    return (X - torch.conj(X).transpose(dim1, dim2)) / 2


def in_lie_algebra(X: torch.Tensor, eps: float = 1e-5) -> bool:
    """
    Determines if the given matrix belongs to the unitary Lie algebra.

    The function checks if the matrix is anti-Hermitian, i.e., it satisfies the condition :math:`A^\dagger = -A`.

    Args:
        X (torch.Tensor): Tensor to check. Must have at least two dimensions where the last two form a square matrix.
        eps (float): Tolerance for numerical comparison. Default is 1e-5.

    Returns:
        bool: True if the matrix belongs to the unitary Lie algebra, False otherwise.

    Raises:
        ValueError: If the input tensor does not have at least 2 dimensions or if the last two dimensions are not square.
    """
    return (
        X.dim() >= 2
        and X.size(-2) == X.size(-1)
        and torch.allclose(torch.conj(X.transpose(-2, -1)), -X, atol=eps)
    )


def initialize_elements(
    tensor: torch.Tensor, distribution_fn: callable = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Initializes the diagonal and upper triangular elements of a tensor, typically for creating an anti-Hermitian matrix.

    The diagonal is initialized with purely imaginary values, and the upper triangular part is split into real and
    imaginary components.

    Args:
        tensor (torch.Tensor): Multi-dimensional tensor where the last two dimensions form a square matrix.
        distribution_fn (callable, optional): Function to initialize the tensor with a specific distribution.
            Defaults to uniform distribution in the range :math:`[-\pi, \pi]`.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - The diagonal initialized with imaginary values.
            - The upper triangular part with real values.
            - The upper triangular part with imaginary values.

    Raises:
        ValueError: If the tensor does not have at least 2 dimensions or if the last two dimensions are not square.
    """
    assert tensor.ndim >= 2 and tensor.size(-1) == tensor.size(
        -2
    ), f"Expected a square matrix in the last two dimensions, got shape {tensor.size()}."

    dim_matrices: int = tensor.size(-2)
    size: int = tensor.size()[:-2]

    diag = tensor.new(size + (dim_matrices,))
    off_diag = tensor.new(size + (2 * dim_matrices, dim_matrices))

    if distribution_fn is None:
        torch.nn.init.uniform_(diag, -math.pi, math.pi)
        torch.nn.init.uniform_(off_diag, -math.pi, math.pi)
    else:
        distribution_fn(diag)
        distribution_fn(off_diag)

    diag = diag.imag * 1j
    upper_tri_real = torch.triu(
        off_diag[..., :dim_matrices, :dim_matrices], 1
    ).real.cfloat()
    upper_tri_complex = (
        torch.triu(off_diag[..., dim_matrices:, :dim_matrices], 1).imag.cfloat() * 1j
    )

    return diag, upper_tri_real, upper_tri_complex


def unitary_lie_init_(
    tensor: torch.Tensor, distribution_fn: callable = None
) -> torch.Tensor:
    r"""
    In-place initialization of a tensor to belong to the unitary Lie algebra.

    The function ensures that the resulting tensor is anti-Hermitian, satisfying :math:`A^\dagger = -A`. The matrix
    is initialized using random distributions, with options to specify a custom distribution function.

    Args:
        tensor (torch.Tensor): Tensor where the last two dimensions form a square matrix.
        distribution_fn (callable, optional): Function to initialize the tensor with a specific distribution.
            Defaults to uniform distribution in the range :math:`[-\pi, \pi]`.

    Returns:
        torch.Tensor: The initialized tensor.

    Raises:
        ValueError: If the initialized tensor does not satisfy the unitary Lie algebra condition after initialization.

    Note:
        Currently, this function modifies the tensor in place. A future enhancement may include returning a new tensor
        without modifying the input tensor.
    """
    diag, upper_tri_real, upper_tri_complex = initialize_elements(
        tensor, distribution_fn
    )

    real_part = (upper_tri_real - upper_tri_real.transpose(-2, -1)) / math.sqrt(2)
    complex_part = (
        upper_tri_complex + upper_tri_complex.transpose(-2, -1)
    ) / math.sqrt(2)

    with torch.no_grad():
        x = real_part + complex_part + torch.diag_embed(diag)
        tensor.copy_(x.cfloat())

        if not in_lie_algebra(x):
            raise ValueError(
                "Initialized tensor does not belong to the unitary Lie algebra. Unknown reason."
            )
    return tensor
