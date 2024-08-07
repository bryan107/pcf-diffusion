import torch


def matrix_power_two_batch(A, k):
    """
    Computes the matrix power of A for each element in k using batch processing.

    Args:
        A (torch.Tensor): Input tensor of shape (..., m, m).
        k (torch.Tensor): Exponent tensor of shape (...).

    Returns:
        torch.Tensor: Resulting tensor of shape (..., m, m).
    """
    orig_size = A.size()
    A, k = A.flatten(0, -3), k.flatten()
    ksorted, idx = torch.sort(k)
    # Abusing bincount...
    count = torch.bincount(ksorted)
    nonzero = torch.nonzero(count, as_tuple=False)
    A = torch.matrix_power(A, 2 ** ksorted[0])
    last = ksorted[0]
    processed = count[nonzero[0]]
    for exp in nonzero[1:]:
        new, last = exp - last, exp
        A[idx[processed:]] = torch.matrix_power(A[idx[processed:]], 2 ** new.item())
        processed += count[exp]
    return A.reshape(orig_size)


def rescale_exp_matrix(f, A):
    """
    Computes the rescaled matrix exponential of A.
    By following formula exp(A) = (exp(A/k))^k

    Args:
        f (callable): Function to compute the matrix exponential.
        A (torch.Tensor): Input tensor of shape (..., m, m).

    Returns:
        torch.Tensor: Resulting tensor of shape (..., m, m).
    """
    normA = torch.max(torch.sum(torch.abs(A), axis=-2), axis=-1).values
    more = normA > 1
    s = torch.ceil(torch.log2(normA)).long()
    s = normA.new_zeros(normA.size(), dtype=torch.long)
    s[more] = torch.ceil(torch.log2(normA[more])).long()
    A_1 = torch.pow(0.5, s.float()).unsqueeze_(-1).unsqueeze_(-1).expand_as(A) * A
    return matrix_power_two_batch(f(A_1), s)
