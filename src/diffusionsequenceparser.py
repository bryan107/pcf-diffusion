from abc import ABC, abstractmethod

import torch


class DiffusionSequenceParser(ABC):
    """
    Base class for parsing sequences for diffusion-based loss computation.
    It is a class of type: typing.Callable[[torch.Tensor], torch.Tensor]

    Methods
    -------
    __call__(self, diffusion_paths: torch.Tensor) -> torch.Tensor:
        Abstract method to modify the input tensor and return the modified sequence.

    validate_input(self, diffusion_paths: torch.Tensor, min_length: int) -> None:
        Validates the diffusion_paths tensor dimensions and sequence length.
    """

    @abstractmethod
    def __call__(self, diffusion_paths: torch.Tensor) -> torch.Tensor:
        pass

    def validate_input(self, diffusion_paths: torch.Tensor, min_length: int) -> None:
        """
        Validate the dimensions of the diffusion_paths tensor.

        Args:
            diffusion_paths (torch.Tensor): A 3D tensor of shape
                (batch_size, sequence_length, feature_dim).
            min_length (int): The minimum number of steps required for this operation.

        Asserts:
            Ensures that the input tensor meets the required dimensions and length.
        """
        assert len(diffusion_paths.shape) == 3, (
            f"Expected 3D tensor (batch_size, sequence_length, feature_dim), "
            f"got {diffusion_paths.shape}."
        )
        assert diffusion_paths.shape[1] >= min_length, (
            f"Sequence length must be at least {min_length}, "
            f"but got {diffusion_paths.shape[1]}."
        )


class TruncationParser(DiffusionSequenceParser):
    """
    Truncates the diffusion sequence to retain only the last `num_steps_to_consider` steps.

    This parser truncates diffusion paths, keeping the last `num_steps_to_consider` steps.
    The quality of the samples at the end of the diffusion process tends to be better, so truncating
    the sequence in this manner helps focus on the best quality steps.

    Note: Ensure that the sequence is in the correct time direction before using this method.

    Args:
        num_steps_to_consider (int): The number of diffusion steps to retain from the end of the sequence.

    .. math::
        T(x) = x_{[-k:]}, \quad k = \text{num_steps_to_consider}
    """

    def __init__(self, num_steps_to_consider: int):
        self.num_steps_to_consider = num_steps_to_consider

    def __call__(self, diffusion_paths: torch.Tensor) -> torch.Tensor:
        """
        Truncate the sequence to retain only the last steps.

        Args:
            diffusion_paths (torch.Tensor): A 3D tensor of shape
                (batch_size, sequence_length, feature_dim).

        Returns:
            torch.Tensor: The truncated diffusion paths tensor.
        """
        self.validate_input(diffusion_paths, self.num_steps_to_consider)
        return diffusion_paths[:, : self.num_steps_to_consider]


class SubsamplingParser(DiffusionSequenceParser):
    """
    Samples a local time `t` and extracts a subsequence of length `local_size` around `t`.

    This parser samples a local time step `t` and then extracts a subsequence of length
    `local_size` from `t - local_size` to `t`.

    Args:
        local_size (int): The number of steps to include around the sampled time `t`.
    """

    def __init__(self, local_size: int):
        self.local_len = local_size

    def __call__(self, diffusion_paths: torch.Tensor) -> torch.Tensor:
        """
        Sample a local time `t` and extract a subsequence around `t`.

        Args:
            diffusion_paths (torch.Tensor): A 3D tensor of shape
                (batch_size, sequence_length, feature_dim).

        Returns:
            torch.Tensor: The subsequence of diffusion paths around the sampled time `t`.
        """
        sequence_length = diffusion_paths.shape[1]
        self.validate_input(diffusion_paths, self.local_len + 1)

        # Sample a valid time t that allows for local_size steps before it
        t_sample = torch.randint(self.local_len, sequence_length, (1,)).item()

        # Return the slice of the tensor from t_sample - local_size to t_sample
        return diffusion_paths[:, t_sample - self.local_len : t_sample]


class UniformGridParser(DiffusionSequenceParser):
    """
    Samples points on a uniform grid along the sequence.

    Args:
        num_points (int): The number of points to sample on the grid.
    """

    def __init__(self, num_points: int):
        self.num_points = num_points

    @staticmethod
    def calculate_grid_indices(sequence_length: int, num_points: int) -> torch.Tensor:
        """
        Calculate uniform grid indices.

        Args:
            sequence_length (int): Length of the sequence.
            num_points (int): Number of points to sample.

        Returns:
            torch.Tensor: Indices sampled uniformly.
        """
        return torch.arange(
            0, sequence_length, step=max(1, sequence_length // num_points)
        )

    def __call__(self, diffusion_paths: torch.Tensor) -> torch.Tensor:
        """
        Sample points on a uniform grid along the diffusion paths.

        Args:
            diffusion_paths (torch.Tensor): 3D tensor of shape
                (batch_size, sequence_length, feature_dim).

        Returns:
            torch.Tensor: The subsequence of diffusion paths sampled on a uniform grid.
        """
        sequence_length = diffusion_paths.shape[1]
        self.validate_input(diffusion_paths, self.num_points)

        indices = self.calculate_grid_indices(sequence_length, self.num_points)
        return diffusion_paths[:, indices]


class RandomPointsParser(DiffusionSequenceParser):
    """
    Samples a random set of points along the sequence.

    Args:
        num_points (int): The number of points to randomly sample along the sequence.
    """

    def __init__(self, num_points: int):
        self.num_points = num_points

    def __call__(self, diffusion_paths: torch.Tensor) -> torch.Tensor:
        """
        Randomly sample a set of points along the diffusion paths.

        Args:
            diffusion_paths (torch.Tensor): 3D tensor of shape
                (batch_size, sequence_length, feature_dim).

        Returns:
            torch.Tensor: The subsequence of diffusion paths with randomly sampled points.
        """
        sequence_length = diffusion_paths.shape[1]
        self.validate_input(diffusion_paths, self.num_points)

        # Randomly sample `num_points` indices
        indices = torch.randperm(sequence_length)[: self.num_points]
        return diffusion_paths[:, indices]
