import typing

import matplotlib.pyplot as plt
import torch
from torch import nn


def histogram_torch(x, bins, density=True):
    """
    Computes the histogram of a tensor using provided bins.

    Args:
    - x (torch.Tensor): Input tensor.
    - bins (torch.Tensor): Precomputed bin edges.
    - density (bool): Whether to normalize the histogram.

    Returns:
    - count (torch.Tensor): Counts of the histogram bins on the device of x.
    - bins (torch.Tensor): Bin edges on the device of x.
    """
    delta = bins[1] - bins[0]
    n_bins = len(bins) - 1

    # Create empty tensor of the right type and size for histogram counts
    count = torch.empty(n_bins, device=x.device, dtype=torch.float32)
    torch.histc(x, bins=n_bins, min=bins[0].item(), max=bins[-1].item(), out=count)

    if density:
        count = count / (delta * x.numel())
    return count, bins


class HistogramLoss(nn.Module):
    # nn.Module because it has a parameter to register.
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

        self.densities, self.center_bin_locs, self.bin_widths, self.bin_edges = self.precompute_histograms(
            x_real, n_bins
        )

        # This list stores the density values of the histograms for each feature at each time step.
        # Each entry in densities corresponds to a particular feature and time step, containing the density values
        # (normalized counts) of the histogram bins.
        self.densities = nn.ParameterList([nn.Parameter(density, requires_grad=False) for density in self.densities])
        # This list stores the locations of the bin centers for each feature at each time step.
        # The bin centers are computed as the midpoints between consecutive bin edges.
        self.center_bin_locs = nn.ParameterList(
            [nn.Parameter(loc, requires_grad=False) for loc in self.center_bin_locs]
        )
        # This list stores the width of the bins (delta) for each feature at each time step.
        # The bin width is the difference between consecutive bin edges.
        self.bin_widths = nn.ParameterList(
            [nn.Parameter(bin_width, requires_grad=False) for bin_width in self.bin_widths]
        )
        # This list stores the bin edges for each feature at each time step.
        # The bin edges define the boundaries of the bins used to compute the histogram.
        # Not used for the loss because we directly check what points are in the bins without recomputing the histogram.
        # Used for plotting nonetheless.
        self.bin_edges = nn.ParameterList([nn.Parameter(bin, requires_grad=False) for bin in self.bin_edges])

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

    def compute(self, x_fake):
        """
        Computes the histogram loss between real and fake data distributions.
        We noticed issues in the case of the comparison of densities ala Dirac measure. Use with cautious in that case.

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

        for time_step in range(x_fake.shape[1]):
            per_time_losses: typing.List = []
            for feature_idx in range(x_fake.shape[2]):
                # Localisation of the bins
                loc: torch.Tensor = self.center_bin_locs[time_step][feature_idx]
                # Fake samples at time step t for feature i
                x_ti: torch.Tensor = x_fake[:, time_step, feature_idx].reshape(-1, 1)
                # Distance bin center to the sample.
                dist: torch.Tensor = torch.abs(x_ti - loc)
                # Counts how many element of the fake data falls within the corresponding bins of the real data.
                counter: torch.Tensor = ((self.bin_widths[time_step][feature_idx] / 2.0 - dist) > 0.0).float()
                # Normalized count of fake data points within each bin
                density: torch.Tensor = counter.mean(0) / self.bin_widths[time_step][feature_idx]
                # Abs difference between the density of the fake data and the density of the real data for each bin
                abs_metric: torch.Tensor = torch.abs(density - self.densities[time_step][feature_idx])
                per_time_losses.append(torch.mean(abs_metric))
            all_losses.append(torch.stack(per_time_losses))
        all_losses: torch.Tensor = torch.stack(all_losses)
        return all_losses

    def forward(self, x_fake, ignore_features: list = None):
        if ignore_features is None:
            return self.compute(x_fake).mean()

        ignore_indices = torch.tensor(ignore_features, dtype=torch.long)
        mask = torch.ones(self.num_features, dtype=torch.bool)
        mask[ignore_indices] = False
        return self.compute(x_fake)[:, mask].mean()

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
                    x_fake[:, time_step, feature_idx], self.bin_edges[time_step][feature_idx], density=True
                )

                # Plot real data histogram
                plt.hist(
                    self.center_bin_locs[time_step][feature_idx].cpu().numpy(),
                    bins=self.center_bin_locs[time_step][feature_idx].cpu().numpy(),
                    weights=self.densities[time_step][feature_idx].cpu().detach().numpy(),
                    alpha=0.5,
                    label='Real Data',
                )

                # Plot fake data histogram using the same bins as real data
                plt.hist(
                    self.center_bin_locs[time_step][feature_idx].cpu().numpy(),
                    bins=self.center_bin_locs[time_step][feature_idx].cpu().numpy(),
                    weights=fake_density.cpu().detach().numpy(),
                    alpha=0.5,
                    label='Fake Data',
                )

                plt.title(f'Feature {feature_idx + 1}, Time Step {time_step + 1}')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.legend()
                plt.pause(0.01)


# Example usage
if __name__ == "__main__":
    N, L, D = 50_000, 5, 2  # Example dimensions
    n_bins = 20

    import time

    start = time.time()

    # Generate random real and fake data
    x_real = torch.randn(N, L, D)  # .to('cuda')
    x_fake = torch.randn(N, L, D) * 0.6 + 0.04  # .to('cuda')

    # Instantiate the HistogramLoss
    histo_loss = HistogramLoss(x_real, n_bins)

    # Compute the loss
    loss = histo_loss(x_fake)
    print("Histogram Loss:", loss)

    end = time.time()
    print("Time taken:", end - start)

    # Plot histograms for debugging
    histo_loss.plot_histograms(x_fake)

    plt.show()
