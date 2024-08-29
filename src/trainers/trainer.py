import logging
import os
import time
import warnings
from collections import defaultdict
from os import path as pt

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from pytorch_lightning import LightningModule

logger = logging.getLogger(__name__)


class Trainer(LightningModule):
    def __init__(
        self,
        test_metrics_train,
        test_metrics_test,
        num_epochs,
        feature_dim_time_series,
        foo=lambda x: x,
    ):
        super().__init__()

        self.num_epochs = num_epochs

        self.losses_history = defaultdict(list)

        self.test_metrics_train = test_metrics_train
        self.test_metrics_test = test_metrics_test
        self.foo = foo

        self.init_time = time.time()

        self.feature_dim_time_series = feature_dim_time_series
        self.plot_samples = plt.subplots(1, 1)[0]
        return

    def evaluate(self, x_fake, x_real, path_file):
        self.losses_history["time"].append(time.time() - self.init_time)

        for i in range(len(self.plot_samples.axes)):
            self.plot_samples.axes[i].clear()

        self.plot_swiss_roll(x_real, x_fake, self.plot_samples, path_file)
        return

    @staticmethod
    def plot_sample_seqs(real_X, fake_X, fig, path_file: str):
        # path file should change if you save multiple times, extension preferably a png.
        # Convention followed is that last axis' last dimension represents time, used for the x-axis.
        # PLots other lines (along second axis for each dimension of the last axis).
        assert (
            real_X.shape[-1] == fake_X.shape[-1]
        ), "Data should have the same sizes, but got {} and {}".format(
            real_X.shape[-1], fake_X.shape[-1]
        )
        assert (
            len(real_X.shape) == 3
        ), "Data should have 3 dimensions, but got {}".format(len(real_X.shape))
        assert (
            len(fake_X.shape) == 3
        ), "Data should have 3 dimensions, but got {}".format(len(fake_X.shape))
        assert len(fig.axes) == real_X.shape[-1] - 1, (
            "Number of subplots should be equal to the number of dimensions of the last axis of the data minus 1, "
            "but got {} and {}"
        ).format(len(fig.axes), real_X.shape[-1] - 1)

        random_indices = torch.randint(real_X.shape[0], (real_X.shape[0],))
        for i in range(real_X.shape[-1] - 1):
            fig.axes[i].plot(
                real_X[random_indices, :, 1].detach().cpu().numpy().T,
                real_X[random_indices, :, 0].detach().cpu().numpy().T,
                "r-x",
                alpha=0.3,
            )

            fig.axes[i].plot(
                fake_X[:, :, 1].detach().cpu().numpy().T,
                fake_X[:, :, 0].detach().cpu().numpy().T,
                "b-x",
                alpha=0.3,
            )

            # Add only one legend entry for each type

            custom_lines = [
                Line2D([0], [0], color="r", marker="x", alpha=0.3, label="real"),
                Line2D([0], [0], color="b", marker="x", alpha=0.3, label="fake"),
            ]

            fig.axes[i].legend(handles=custom_lines)

        directory_where_to_save = os.path.dirname(path_file)
        if not os.path.exists(directory_where_to_save):
            if directory_where_to_save != "":
                os.makedirs(directory_where_to_save)
        fig.savefig(pt.join(path_file))

        plt.pause(0.01)
        return

    @staticmethod
    def plot_swiss_roll(real_X, fake_X, fig, path_file: str):
        # path file should change if you save multiple times, extension preferably a png.
        # Convention followed is that last axis' last dimension represents time, used for the x-axis.
        # PLots other lines (along second axis for each dimension of the last axis).
        assert (
            real_X.shape[-1] == fake_X.shape[-1]
        ), "Data should have the same sizes, but got {} and {}".format(
            real_X.shape[-1], fake_X.shape[-1]
        )
        assert (
            len(real_X.shape) == 3
        ), "Data should have 3 dimensions, but got {}".format(len(real_X.shape))
        assert (
            len(fake_X.shape) == 3
        ), "Data should have 3 dimensions, but got {}".format(len(fake_X.shape))

        random_indices = torch.randint(real_X.shape[0], (real_X.shape[0],))

        # Only supporting 2D
        if real_X.shape[-1] != 3:
            # warning without terminating

            warnings.warn(
                "Only supporting 2D data for swiss roll! So showing 2 out of 3 dimensions. Here, we received {} dimensions.".format(
                    real_X.shape[-1]
                ),
                RuntimeWarning,
            )
            logger.warn(
                "Only supporting 2D data for swiss roll! So showing 2 out of 3 dimensions. Here, we received {} dimensions.".format(
                    real_X.shape[-1]
                )
            )

        plt.scatter(
            real_X[random_indices, -1, 0].detach().cpu().numpy().T,
            real_X[random_indices, -1, 1].detach().cpu().numpy().T,
            alpha=0.5,
        )
        plt.scatter(
            fake_X[:, 0].detach().cpu().numpy().T,
            fake_X[:, 1].detach().cpu().numpy().T,
            marker="1",
            alpha=0.5,
        )
        plt.legend(["Original data", "Generated data"])

        directory_where_to_save = os.path.dirname(path_file)
        if not os.path.exists(directory_where_to_save):
            if directory_where_to_save != "":
                os.makedirs(directory_where_to_save)
        fig.savefig(pt.join(path_file))

        plt.pause(0.01)
        return
