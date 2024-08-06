import time
from collections import defaultdict
from os import path as pt

import matplotlib.pyplot as plt
import seaborn as sns
import torch


class Trainer:
    def __init__(
        self,
        batch_size,
        G,
        G_optimizer,
        test_metrics_train,
        test_metrics_test,
        n_gradient_steps,
        foo=lambda x: x,
    ):
        self.batch_size = batch_size

        self.G = G
        self.G_optimizer = G_optimizer
        self.n_gradient_steps = n_gradient_steps

        self.losses_history = defaultdict(list)

        self.test_metrics_train = test_metrics_train
        self.test_metrics_test = test_metrics_test
        self.foo = foo

        self.init_time = time.time()

    def evaluate(self, x_fake, x_real, step, config, **kwargs):
        self.losses_history["time"].append(time.time() - self.init_time)

        self.plot_sample_seqs(x_real, x_fake[: config.batch_size], self.config, step)
        return

    @staticmethod
    def plot_sample_seqs(real_X, fake_X, config, step):
        plt.close()
        plt.figure()

        random_indices = torch.randint(0, real_X.shape[0], (config.batch_size,))
        plt.plot(
            real_X[random_indices, :, 1].detach().cpu().numpy().T,
            real_X[random_indices, :, 0].detach().cpu().numpy().T,
            "r-x",
            alpha=0.3,
        )

        plt.plot(
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

        plt.legend(handles=custom_lines)
        plt.savefig(pt.join(config.exp_dir, "x_" + str(step) + ".png"))
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
