import os
import time
from collections import defaultdict
from os import path as pt

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pytorch_lightning import LightningModule


class Trainer(LightningModule):
    def __init__(
        self,
        test_metrics_train,
        test_metrics_test,
        num_epochs,
        foo=lambda x: x,
    ):
        super().__init__()

        self.num_epochs = num_epochs

        self.losses_history = defaultdict(list)

        self.test_metrics_train = test_metrics_train
        self.test_metrics_test = test_metrics_test
        self.foo = foo

        self.init_time = time.time()

        self.plot_samples = plt.subplots(1, 1)[0]
        return

    def evaluate(self, x_fake, x_real, path_file):
        self.losses_history["time"].append(time.time() - self.init_time)
        self.plot_samples.axes[0].clear()

        self.plot_sample_seqs(x_real, x_fake, self.plot_samples, path_file)
        return

    @staticmethod
    def plot_sample_seqs(real_X, fake_X, fig, path_file: str):
        # path file should change if you save multiple times, extension preferably a png.

        random_indices = torch.randint(real_X.shape[0], (real_X.shape[0],))
        fig.axes[0].plot(
            real_X[random_indices, :, 1].detach().cpu().numpy().T,
            real_X[random_indices, :, 0].detach().cpu().numpy().T,
            "r-x",
            alpha=0.3,
        )

        fig.axes[0].plot(
            fake_X[:, :, 1].detach().cpu().numpy().T,
            fake_X[:, :, 0].detach().cpu().numpy().T,
            "b-x",
            alpha=0.3,
        )

        # Add only one legend entry for each type
        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color="r", marker="x", alpha=0.3, label="real"),
            Line2D([0], [0], color="b", marker="x", alpha=0.3, label="fake"),
        ]

        fig.axes[0].legend(handles=custom_lines)

        directory_where_to_save = os.path.dirname(path_file)
        if not os.path.exists(directory_where_to_save):
            if directory_where_to_save != "":
                os.makedirs(directory_where_to_save)
        fig.savefig(pt.join(path_file))

        plt.pause(0.01)
        return

    @staticmethod
    ###@deprecated("FROM THE REPO")
    def plot_reconstructed_sample(
        real_X: torch.tensor, rec_X: torch.tensor, config, step
    ):
        sns.set()
        fig, axs = plt.subplots(1, 2)
        x_real_dim = real_X.shape[-1]
        for i in range(x_real_dim):
            axs[0].plot(real_X[:, i].detach().cpu().numpy().T)
        for i in range(x_real_dim):
            axs[1].plot(rec_X[:, i].detach().cpu().numpy().T)
        plt.savefig(
            pt.join(config.exp_dir, "reconstruction_sample_" + str(step) + ".png")
        )
        plt.close()
