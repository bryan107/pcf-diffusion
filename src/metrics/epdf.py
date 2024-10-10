import logging
import math
import typing

import matplotlib.pyplot as plt
import torch
from torch import nn

logger = logging.getLogger(__name__)


def histogram_torch(x, bins, density=True):
    """
    Computes the histogram of a tensor using provided bins, ignoring NaNs.

    Args:
    - x (torch.Tensor): Input tensor.
    - bins (torch.Tensor): Precomputed bin edges.
    - density (bool): Whether to normalize the histogram.

    Returns:
    - count (torch.Tensor): Counts of the histogram bins on the device of x.
    - bins (torch.Tensor): Bin edges on the device of x.
    """
    # Remove NaNs from the input tensor
    x = x[~torch.isnan(x)]
    # Return zero count if no valid entries remain
    if x.numel() == 0:
        logger.warning(
            "There are only NaNs in the input tensor for histogram computation."
        )
        return torch.zeros(len(bins) - 1, device=x.device), bins

    delta = bins[1] - bins[0]
    n_bins = len(bins) - 1

    # Create empty tensor of the right type and size for histogram counts
    count = torch.empty(n_bins, device=x.device, dtype=torch.float32)
    torch.histc(x, bins=n_bins, min=bins[0].item(), max=bins[-1].item(), out=count)

    if density:
        count = count / (delta * x.numel())
    return count, bins


class HistogramLoss(nn.Module):
    r"""
    HistogramLoss computes the difference between real and fake data distributions
    based on histograms.

    Formulas used for the loss computation:

    1. **Bin Centers**:

    .. math::
        \text{center\_bin\_loc} = \frac{bins[i+1] + bins[i]}{2}

    where `bins[i]` are the edges of the bins.

    2. **Density of Fake Data**:

    .. math::
        \text{density} = \frac{1}{|\Delta|} \cdot \frac{1}{N} \sum_{j=1}^N I\left(\frac{|\text{fake\_sample} - \text{center\_bin\_loc}|}{|\Delta|/2} \leq 1 \right)

    where `\Delta` is the bin width, and the indicator function ensures that the
    fake sample is within the bin centered at `center_bin_loc`.

    3. **Loss**:

    .. math::
        \text{loss} = \frac{1}{|\Delta|} \cdot | \text{density}_{real} - \text{density}_{fake} |

    This computes the absolute difference between the real and fake densities,
    normalized by the bin width `\Delta`.
    """

    @staticmethod
    def num_bins_freedman_diaconis_rule(num_samples):
        """
        Calculate the number of bins using the Freedman-Diaconis rule.

        The Freedman-Diaconis rule suggests the optimal number of bins to use in a histogram
        by minimizing the variance of the histogram.

        The number of bins is computed as:

        .. math::

            bins = 2 * n^{1/3}

        where:
            - n is the number of samples.

        Args:
            num_samples (int): The number of samples in the dataset.

        Returns:
            int: The computed number of bins based on the Freedman-Diaconis rule.
        """
        return int(round(2.0 * math.pow(num_samples, 1.0 / 3.0), 0))

    @staticmethod
    def precompute_histograms(x: torch.Tensor, n_bins: int):
        densities: typing.List = []
        center_bin_locs: typing.List = []
        bin_widths: typing.List = []
        bin_edges: typing.List = []

        for time_step in range(x.shape[1]):
            per_time_densities = []
            per_time_center_bin_locs = []
            per_time_bin_widths = []
            feature_bins = []
            for feature_idx in range(x.shape[2]):
                x_ti = x[:, time_step, feature_idx].reshape(-1)
                x_ti = x_ti[~torch.isnan(x_ti)]  # Remove NaNs

                if x_ti.numel() == 0:
                    # Handle the case where all values are NaNs by appending a tensor of zeros
                    per_time_densities.append(torch.zeros(n_bins, device=x.device))
                    per_time_center_bin_locs.append(
                        torch.zeros(n_bins, device=x.device)
                    )
                    per_time_bin_widths.append(
                        torch.tensor(1.0, device=x.device)
                    )  # Default bin width
                    feature_bins.append(torch.zeros(n_bins + 1, device=x.device))
                    continue

                min_val, max_val = x_ti.min().item(), x_ti.max().item()
                # We catch here the case when the values are all the same for a time and feature.
                if abs(max_val - min_val) < 1e-10:
                    max_val = max_val + 1e-5
                    min_val = min_val - 1e-5

                bins = torch.linspace(min_val, max_val, n_bins + 1, device=x.device)
                density, bins = histogram_torch(x_ti, bins, density=True)
                per_time_densities.append(density)
                bin_width = bins[1] - bins[0]
                center_bin_loc = 0.5 * (bins[1:] + bins[:-1])
                per_time_center_bin_locs.append(center_bin_loc)
                per_time_bin_widths.append(bin_width)
                feature_bins.append(bins)

            densities.append(per_time_densities)
            center_bin_locs.append(per_time_center_bin_locs)
            bin_widths.append(per_time_bin_widths)
            bin_edges.append(feature_bins)

        # For all time stamps, they should be the same dimensions, hence stackable.
        # Can't do ParamList of ParamList. First nest per feature, second per time and inside per bin.
        densities: typing.List = [torch.stack(d) for d in densities]
        center_bin_locs: typing.List = [torch.stack(l) for l in center_bin_locs]
        bin_widths: typing.List = [torch.stack(d) for d in bin_widths]
        bin_edges: typing.List = [torch.stack(b) for b in bin_edges]

        return densities, center_bin_locs, bin_widths, bin_edges

    def __init__(self, x_real: torch.Tensor, n_bins: int):
        """
        Initializes the HistogramLoss with the real data distribution.

        Args:
        - x_real (torch.Tensor): Real data tensor of shape (N, L, D).
        - n_bins (int): Number of bins for the histograms.
        """
        super().__init__()
        self.n_bins = n_bins
        self.num_samples, self.num_time_steps, self.num_features = x_real.shape

        # Log the initialization details
        logger.info(
            f"Initializing HistogramLoss with {self.num_samples} samples, {self.num_time_steps} time steps, and {self.num_features} features."
        )

        self.densities, self.center_bin_locs, self.bin_widths, self.bin_edges = (
            self.precompute_histograms(x_real, n_bins)
        )

        # This list stores the density values of the histograms for each feature at each time step.
        # Each entry in densities corresponds to a particular feature and time step, containing the density values
        # (normalized counts) of the histogram bins.
        self.densities = nn.ParameterList(
            [nn.Parameter(density, requires_grad=False) for density in self.densities]
        )
        # This list stores the locations of the bin centers for each feature at each time step.
        # The bin centers are computed as the midpoints between consecutive bin edges.
        self.center_bin_locs = nn.ParameterList(
            [nn.Parameter(loc, requires_grad=False) for loc in self.center_bin_locs]
        )
        # This list stores the width of the bins (delta) for each feature at each time step.
        # The bin width is the difference between consecutive bin edges.
        self.bin_widths = nn.ParameterList(
            [
                nn.Parameter(bin_width, requires_grad=False)
                for bin_width in self.bin_widths
            ]
        )
        # This list stores the bin edges for each feature at each time step.
        # The bin edges define the boundaries of the bins used to compute the histogram.
        # Not used for the loss because we directly check what points are in the bins without recomputing the histogram.
        # Used for plotting nonetheless.
        self.bin_edges = nn.ParameterList(
            [nn.Parameter(bin, requires_grad=False) for bin in self.bin_edges]
        )

    def compute(self, x_fake):
        """
        Computes the histogram loss between real and fake data distributions.
        We noticed issues in the case of the comparison of densities ala Dirac measure. Use with caution in that case.

        Args:
        - x_fake (torch.Tensor): Fake data tensor of shape (N, L, D).

        Returns:
        - all_losses (torch.Tensor): Component-wise loss. Shape (L, D), representing loss per time per feature.
        """
        assert (
            x_fake.shape[2] == self.num_features
        ), f"Expected {self.num_features} features in x_fake, but got {x_fake.shape[2]}."
        assert (
            x_fake.shape[1] == self.num_time_steps
        ), f"Expected {self.num_time_steps} time steps in x_fake, but got {x_fake.shape[1]}."

        all_losses: typing.List = []
        # To store time steps with NaNs
        nan_features_per_time_step: typing.List[typing.Tuple[int, typing.List[int]]] = (
            []
        )

        for time_step in range(x_fake.shape[1]):
            per_time_losses: typing.List = []
            nan_features: typing.List[int] = (
                []
            )  # Collect indices with NaNs for this time step
            for feature_idx in range(x_fake.shape[2]):
                # Localisation of the bins
                loc: torch.Tensor = self.center_bin_locs[time_step][feature_idx]
                # Fake samples at time step t for feature i
                x_ti: torch.Tensor = x_fake[:, time_step, feature_idx].reshape(-1, 1)
                nan_indices = torch.isnan(x_fake[:, time_step, feature_idx])

                if torch.all(nan_indices):
                    nan_features.append(
                        feature_idx
                    )  # Record feature index with all NaNs
                    continue  # Skip if all NaNs

                x_ti = x_ti[~torch.isnan(x_ti)].reshape(-1, 1)  # Remove NaNs
                if x_ti.numel() == 0:
                    continue  # Skip if no valid entries after removing NaNs

                # Distance bin center to the sample.
                dist: torch.Tensor = torch.abs(x_ti - loc)
                # Counts how many element of the fake data falls within the corresponding bins of the real data.
                counter: torch.Tensor = (
                    (self.bin_widths[time_step][feature_idx] / 2.0 - dist) > 0.0
                ).float()
                # Normalized count of fake data points within each bin
                density: torch.Tensor = (
                    counter.mean(0) / self.bin_widths[time_step][feature_idx]
                )
                # Abs difference between the density of the fake data and the density of the real data for each bin
                abs_metric: torch.Tensor = torch.abs(
                    density - self.densities[time_step][feature_idx]
                )
                per_time_losses.append(torch.mean(abs_metric))

            if not per_time_losses:
                nan_features_per_time_step.append(
                    (time_step, nan_features)
                )  # Store if all features are NaNs for this step
                continue  # Skip if no valid data for this time step

            all_losses.append(torch.stack(per_time_losses))

        # Log NaNs at the end of processing
        if nan_features_per_time_step:
            nan_warnings = [
                f"Time step {time_step} has only NaNs for features {features}"
                for time_step, features in nan_features_per_time_step
            ]
            nan_warnings = nan_warnings[:10] + (
                ["..."] if len(nan_warnings) > 5 else []
            )
            logger.warning(", ".join(nan_warnings))

        # Raise error if no valid data was found for any time step and feature
        if not all_losses:
            logger.error(
                "All time steps and features contain NaNs or empty data, yielding no valid losses."
            )

        all_losses: torch.Tensor = torch.stack(all_losses)
        return all_losses

    def forward(self, x_fake, ignore_features: list = None):
        try:
            if ignore_features is None or (
                hasattr(ignore_features, "__len__") and len(ignore_features) == 0
            ):
                return self.compute(x_fake).mean()

            ignore_indices = torch.tensor(ignore_features, dtype=torch.long)
            mask = torch.ones(x_fake.shape[2], dtype=torch.bool)
            mask[ignore_indices] = False
            return self.compute(x_fake)[:, mask].mean()
        except Exception as e:
            logger.error(f"Error in the forward pass of the HistogramLoss: {e}")
            return torch.tensor(0.0, device=x_fake.device)

    def plot_histograms(self, x_fake):
        """
        Plots histograms for real and fake data for each feature at each time step.

        Args:
        - x_fake (torch.Tensor): Fake data tensor of shape (N, L, D).
        """
        for time_step in range(x_fake.shape[1]):
            for feature_idx in range(x_fake.shape[2]):

                plt.figure(figsize=(10, 5))

                # Compute histogram for fake data using the same bins as real data
                fake_density, _ = histogram_torch(
                    x_fake[:, time_step, feature_idx],
                    self.bin_edges[time_step][feature_idx],
                    density=True,
                )

                # Plot real data histogram
                plt.hist(
                    self.center_bin_locs[time_step][feature_idx].cpu().numpy(),
                    bins=self.center_bin_locs[time_step][feature_idx].cpu().numpy(),
                    weights=self.densities[time_step][feature_idx]
                    .cpu()
                    .detach()
                    .numpy(),
                    alpha=0.5,
                    label="Real Data",
                )

                # Plot fake data histogram using the same bins as real data
                plt.hist(
                    self.center_bin_locs[time_step][feature_idx].cpu().numpy(),
                    bins=self.center_bin_locs[time_step][feature_idx].cpu().numpy(),
                    weights=fake_density.cpu().detach().numpy(),
                    alpha=0.5,
                    label="Fake Data",
                )

                plt.title(f"Feature {feature_idx + 1}, Time Step {time_step + 1}")
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.legend()
                plt.pause(0.01)


if __name__ == "__main__":
    N, L, D = 50_000, 3, 2  # Example dimensions
    n_bins = 20

    # --------------------- EXAMPLE 1: Everything works (no NaNs) ---------------------
    print("Example 1: Basic case with no NaNs")

    # Generate random real and fake data
    x_real = torch.randn(N, L, D)
    x_fake = torch.randn(N, L, D) * 0.6 + 0.04

    # Instantiate the HistogramLoss
    histo_loss = HistogramLoss(x_real, n_bins)

    # Compute the loss
    loss = histo_loss(x_fake)
    print("Histogram Loss (Example 1):", loss)

    # Plot histograms for debugging
    histo_loss.plot_histograms(x_fake)
    plt.pause(0.01)

    # --------------------- EXAMPLE 2: Running on CUDA ---------------------
    if torch.cuda.is_available():
        print("\nExample 2: Running on CUDA")

        # Generate random real and fake data on CUDA
        x_real_cuda = torch.randn(N, L, D).cuda()
        x_fake_cuda = (torch.randn(N, L, D) * 0.6 + 0.04).cuda()

        # Instantiate the HistogramLoss on CUDA
        histo_loss_cuda = HistogramLoss(x_real_cuda, n_bins).cuda()

        # Compute the loss on CUDA
        loss_cuda = histo_loss_cuda(x_fake_cuda)
        print("Histogram Loss (Example 2 - CUDA):", loss_cuda)

        # Plot histograms for debugging
        histo_loss_cuda.to("cpu")
        histo_loss_cuda.plot_histograms(
            x_fake_cuda.to("cpu")
        )  # Move to CPU for plotting
        plt.pause(0.01)
    else:
        print("CUDA is not available. Skipping Example 2.")

    # --------------------- EXAMPLE 3: Handling NaNs ---------------------
    print("\nExample 3: Handling NaNs in the dataset")

    # Generate random real and fake data with some NaNs
    x_real_nans = torch.randn(N, L, D)
    x_fake_nans = torch.randn(N, L, D) * 0.6 + 0.04

    # Introduce NaNs in real and fake data
    x_real_nans[0:1000, 1, 0] = float("nan")  # Feature 1 at time step 2 has some NaNs
    x_fake_nans[100:500, 0, 0] = float("nan")  # Feature 1 at time step 1

    # Instantiate the HistogramLoss
    histo_loss_nans = HistogramLoss(x_real_nans, n_bins)

    # Compute the loss
    loss_nans = histo_loss_nans(x_fake_nans)
    print("Histogram Loss (Example 3 - NaNs):", loss_nans)

    # Plot histograms for debugging
    histo_loss_nans.plot_histograms(x_fake_nans)
    plt.pause(0.01)

    # --------------------- EXAMPLE 4: Handling NaNs ---------------------
    print("\nExample 4: Handling an entire row of NaNs in the dataset")

    # Generate random real and fake data with some NaNs
    x_real_nans = torch.randn(N, L, D)
    x_fake_nans = torch.randn(N, L, D) * 0.6 + 0.04

    # Introduce NaNs in real and fake data
    x_real_nans[0:1000, 1, 0] = float("nan")  # Feature 1 at time step 2 has some NaNs
    x_real_nans[:, 2, 1] = float("nan")  # Feature 2 at time step 3 is entirely NaN
    x_fake_nans[100:500, 0, 0] = float("nan")  # Feature 1 at time step 1

    # Instantiate the HistogramLoss
    histo_loss_nans = HistogramLoss(x_real_nans, n_bins)

    # Compute the loss
    loss_nans = histo_loss_nans(x_fake_nans)
    print("Histogram Loss (Example 4 - NaNs):", loss_nans)

    # Plot histograms for debugging
    histo_loss_nans.plot_histograms(x_fake_nans)
    plt.show()
