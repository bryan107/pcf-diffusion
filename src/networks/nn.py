import torch


def matrix_power_two_batch(matrix, exponents):
    """
    Computes the matrix power of a tensor using batch processing.

    Given a tensor ``matrix`` of shape ``(..., m, m)`` and a tensor ``exponents`` of shape ``(...)``,
    this function computes :math:`matrix^{2^k}` efficiently using batch processing.

    Args:
        matrix (torch.Tensor): Input tensor of shape ``(..., m, m)``, where ``m`` is the dimension of the square matrices.
        exponents (torch.Tensor): Exponent tensor (ideally 1D), containing the exponents for the matrix powers.

    Returns:
        torch.Tensor: Tensor of shape ``(..., m, m)`` where each matrix in the batch has been raised
        to the corresponding power :math:`2^k`.

    Explanation:
        1. The input tensors ``matrix`` and ``exponents`` are flattened to allow for efficient batch processing.
        2. The exponents tensor ``exponents`` is sorted, and the matrix power is computed iteratively.
        3. Using the sorted exponents tensor, the function calculates powers of the matrices efficiently by exploiting
           the property :math:`matrix^{2^k} = (matrix^{2^{k-1}})^2`.
        4. Finally, the resulting matrices are reshaped back to their original batch shape.

    Example:
        If ``matrix`` is a batch of 2x2 matrices and ``exponents`` contains the exponents, this function will compute
        each matrix raised to the power :math:`2^k` for ``k`` in ``exponents`` using a more efficient strategy than direct
        computation.

    .. autofunction:: matrix_power_two_batch
    """
    tensor_shape = matrix.size()
    matrix, exponents = matrix.flatten(0, -3), exponents.flatten()
    ordered_exponents, idx = torch.sort(exponents)
    count = torch.bincount(ordered_exponents)
    nonzero = torch.nonzero(count, as_tuple=False)
    matrix = torch.matrix_power(matrix, 2 ** ordered_exponents[0])
    last = ordered_exponents[0]
    processed = count[nonzero[0]]
    for exp in nonzero[1:]:
        new, last = exp - last, exp
        matrix[idx[processed:]] = torch.matrix_power(
            matrix[idx[processed:]], 2 ** new.item()
        )
        processed += count[exp]
    return matrix.reshape(tensor_shape)


def rescale_exp_matrix(f, A):
    """
    Computes the rescaled matrix exponential using the scaling and squaring method.

    The matrix exponential :math:`\exp(A)` is computed by first scaling the matrix by a factor of :math:`2^{-s}`,
    where :math:`s` is chosen such that the scaled matrix has a norm less than or equal to 1.
    Then, the exponential of the scaled matrix is raised to the power of :math:`2^s` to obtain the result.

    Mathematically, the method follows the formula:

    .. math::

        \exp(A) = \left( \exp\left(\frac{A}{2^s}\right) \right)^{2^s}

    Args:
        f (callable): A function that computes the matrix exponential, such as `torch.linalg.matrix_exp`.
        A (torch.Tensor): Input tensor of shape ``(..., m, m)``, where ``m`` is the dimension of the square matrices.

    Returns:
        torch.Tensor: The resulting tensor of shape ``(..., m, m)`` after applying the rescaled matrix exponential.

    Explanation:
        1. The norm of each matrix is computed using the maximum sum of absolute values of each row.
        2. The scaling factor :math:`s` is computed as :math:`s = \lceil \log_2(\|A\|) \rceil` for matrices with norm greater than 1.
        3. The matrix :math:`A` is then scaled by :math:`2^{-s}` to ensure that its norm is reduced.
        4. The matrix exponential of the scaled matrix is computed using the provided function ``f``.
        5. Finally, the scaled matrix exponential is raised to the power :math:`2^s` using ``matrix_power_two_batch`` to obtain the final result.

    Example:
        This method is particularly useful for computing the matrix exponential for matrices with large norms,
        as it improves numerical stability.

    .. autofunction:: rescale_exp_matrix
    """
    normA = torch.max(torch.sum(torch.abs(A), axis=-2), axis=-1).values
    more = normA > 1
    s = torch.ceil(torch.log2(normA)).long()
    s = normA.new_zeros(normA.size(), dtype=torch.long)
    s[more] = torch.ceil(torch.log2(normA[more])).long()
    A_1 = torch.pow(0.5, s.float()).unsqueeze_(-1).unsqueeze_(-1).expand_as(A) * A
    return matrix_power_two_batch(f(A_1), s)
