import enum
import logging
import warnings
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D

from src.utils.utils_os import savefig

logger = logging.getLogger(__name__)


class DataType(enum.Enum):
    ONE_D = 1
    TWO_D = 2
    THREE_D = 3
    TIME_SERIES = 4
    IMAGES = 5

    @property
    def plot_method(self):
        plot_methods = {
            DataType.ONE_D: plot_histograms,
            DataType.TWO_D: plot_2D_dataset,
            DataType.TIME_SERIES: plot_sample_seqs,
        }
        return plot_methods.get(self)


def plot_histograms(
    targets: torch.Tensor,
    samples: torch.Tensor,
    axes: List[plt.Axes],
    path_file: Optional[str],
    legend: bool,
):
    assert (
        targets.shape[-1] == samples.shape[-1]
    ), "Data should have the same number of features, but got {} and {}".format(
        targets.shape[-1], samples.shape[-1]
    )
    assert len(targets.shape) == 2, "Data should have 2 dimensions, but got {}".format(
        len(targets.shape)
    )
    assert len(samples.shape) == 2, "Data should have 2 dimensions, but got {}".format(
        len(samples.shape)
    )
    assert (
        len(axes) == targets.shape[-1]
    ), f"Expected {targets.shape[-1]} axes, but got {len(axes)}"

    for i, ax in enumerate(axes):
        sns.distplot(
            targets[:, i].detach().cpu().numpy(),
            kde=True,
            color="blue",
            label="Real Data",
            hist=True,
            ax=ax,
            bins=len(targets[:, i]) // 10,
        )

        sns.distplot(
            samples[:, i].detach().cpu().numpy(),
            kde=True,
            color="red",
            label="Sampled Data",
            hist=True,
            ax=ax,
            bins=len(targets[:, i]) // 10,
        )

        if legend:
            ax.set_title(
                f"Histogram with KDE comparing true (n={targets.shape[0]}) and generated (n={samples.shape[0]}) data"
            )
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()

    if path_file is not None:
        savefig(axes[0].figure, path_file)
    return


##### NOT TESTED
def plot_sample_seqs(
    targets: torch.Tensor,
    samples: torch.Tensor,
    axes: List[plt.Axes],
    path_file: Optional[str],
    legend: bool,
):
    # path file should change if you save multiple times, extension preferably a png.
    # Convention followed is that last axis' last dimension represents time, used for the x-axis.
    # PLots other lines (along second axis for each dimension of the last axis).
    assert (
        targets.shape[-1] == samples.shape[-1]
    ), f"Data should have the same number of features, but got {targets.shape[-1]} and {samples.shape[-1]}"
    assert (
        len(targets.shape) == 2
    ), f"Data should have 2 dimensions, but got {len(targets.shape)}"
    assert (
        len(samples.shape) == 2
    ), f"Data should have 2 dimensions, but got {len(samples.shape)}"
    assert (
        len(axes) == targets.shape[-1]
    ), f"Expected {targets.shape[-1]} axes, but got {len(axes)}"

    random_indices = torch.randint(targets.shape[0], (targets.shape[0],))
    for i, ax in enumerate(axes):
        ax.plot(
            np.arange(targets.shape[1]),
            targets[random_indices, :, i].detach().cpu().numpy().T,
            "r-x",
            alpha=0.3,
        )

        ax.plot(
            np.arange(samples.shape[1]),
            samples[:, :, i].detach().cpu().numpy().T,
            "b-x",
            alpha=0.3,
        )

        # Add only one legend entry for each type
        custom_lines = [
            Line2D([0], [0], color="r", marker="x", alpha=0.3, label="real"),
            Line2D([0], [0], color="b", marker="x", alpha=0.3, label="fake"),
        ]

        if legend:
            ax.legend(handles=custom_lines)

    savefig(axes[0].figure, path_file)
    return


def plot_2D_dataset(
    targets: torch.Tensor,
    samples: torch.Tensor,
    axes: List[plt.Axes],
    path_file: Optional[str],
    legend: bool,
):
    # path file should change if you save multiple times, extension preferably a png.
    # Convention followed is that last axis' last dimension represents time, used for the x-axis.
    # PLots other lines (along second axis for each dimension of the last axis).
    assert (
        targets.shape[-1] == samples.shape[-1]
    ), "Data should have the same sizes, but got {} and {}".format(
        targets.shape[-1], samples.shape[-1]
    )
    assert len(targets.shape) == 2, "Data should have 2 dimensions, but got {}".format(
        len(targets.shape)
    )
    assert len(samples.shape) == 2, "Data should have 2 dimensions, but got {}".format(
        len(samples.shape)
    )
    assert len(axes) == 1, "Expected one axis for plotting, but got none."

    random_indices = torch.randint(targets.shape[0], (targets.shape[0],))

    # Only supporting 2D
    if targets.shape[-1] != 2:
        warnings.warn(
            "Only supporting 2D data for swiss roll! So showing 2 out of 3 dimensions. Here, we received {} dimensions.".format(
                targets.shape[-1]
            ),
            RuntimeWarning,
        )

    axes[0].scatter(
        targets[random_indices, 0].detach().cpu().numpy().T,
        targets[random_indices, 1].detach().cpu().numpy().T,
        alpha=0.5,
    )
    axes[0].scatter(
        samples[:, 0].detach().cpu().numpy().T,
        samples[:, 1].detach().cpu().numpy().T,
        marker="1",
        alpha=0.5,
    )
    if legend:
        axes[0].legend(["Original data", "Generated data"])

    if path_file is not None:
        savefig(axes[0].figure, path_file)
    return
