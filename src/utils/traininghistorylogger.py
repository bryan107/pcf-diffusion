import typing

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


# Document the two phenomena:
# - The validation is logged before the backpropagation step, unless it is manually performed. Hence, we should have a shift depending on that.
# For example, imagine we log every 50 epochs. At epoch 49, the validation loss is the same as the validation loss at epoch 0. So, we are logging at epoch 50 the validation of epoch 0!
# This is only if we are not manually back-propagating, in which case, the validation loss is for the epoch 49. Then, at epoch 50, the new validation loss is calculated.
# - The early stoppers are called before the training of the current step (at epoch 48 in the period=50 example) and after validation of the current step happening before the training of the current step.
# So, in order to log the same loss as used for the best model choice, it is better to early stop on validation losses, which reflect the current state. Otherwise, one observes a shift of 1 epoch.
# Ref: https://github.com/Lightning-AI/pytorch-lightning/issues/1464


### If too many metrics are requested, we do not plot at all!
class TrainingHistoryLogger(LightningLoggerBase):
    """
    Useful class that is at the same time:
        - helper to save the history of training somewhere on the heap instead of in a file.
        - helper to plot the evolution of the losses DURING training,
        - adaptor from a dictionary with results to estim_history class,
        - stores the hyperparameters of the model.

    There is no particular rule to use it.
    However, we noticed that validation metrics are shifted compared to training.
    We provide a solution, whenever the name of the validation metrics contain the words 'val' or 'validation'.
    It is encouraged to do include these words in the naming of the metric.
    We also recommend writing the names of the metrics as: name_type where type can be training or validation.
    This is useful when one converts the history dict into an estimator history.

    TODO : allow for metrics logged at different epochs.
    """

    # for this class to work properly, when one logs from the validation_step,
    # it should contain in the string (name) one sub-string from the list below.
    val_keywords = ["val", "validation"]

    def __init__(
        self,
        metrics: typing.Iterable[str],
        plot_loss_history: bool = False,
        frequency_epoch_logging: int = 1,
    ):
        # frequency_epoch_logging = check_val_every_n_epoch from trainer.
        # metrics are the metrics that will be plotted at each iteration of the evolution of the loss. Iterable.
        # It also contains the metrics that are stored. If the metrics name do not agree with what is logged,
        # there will be a bug.
        super().__init__()

        self.hyper_params = (
            {}
        )  # by default a dictionary because it is the way it is stored.
        self.history = {}
        # The defaultdict will create an entry with an empty list if they key is missing when trying to access
        self.freq_epch = frequency_epoch_logging
        if plot_loss_history:
            self.fig, self.ax = plt.subplots(figsize=(7, 5))  # Create a figure and axis
            self.colors = sns.color_palette("Dark2")  # Get a color palette from seaborn
        else:
            self.fig, self.ax = None, None

        self.metrics: typing.Iterable[str] = metrics
        # Adding a nan to the metrics with validation inside
        # (this is because pytorch lightning does validation after the optimisation step).
        # Hence, there is always a shift between the two values.
        self.history["epoch"] = []
        for name in self.metrics:
            # This is done such that we can use fetch_score method in the plotting method.
            if any(
                val_keyword in name
                for val_keyword in TrainingHistoryLogger.val_keywords
            ):
                self.history[name] = [np.NAN]
            else:
                self.history[name] = []

    @rank_zero_only
    def log_metrics(self, metrics: typing.Dict, step: int):
        """
        Args:
            metrics (dictionary): contains at least the metric name (one metric), the value and the epoch nb.
            Looks like: {'epoch': 0, 'train_metric1': 2.0, 'train_metric2': 1.0}
            step: Period of logging. (validation or training).
            Should match `check_val_every_n_epoch` from the trainer!

        """
        # The trainer from pytorch lightning logs every check_val_every_n_epoch starting from check_val_every_n_epoch -1.
        # So we account for that shift with the + 1.
        if ((metrics["epoch"] + 1) % self.freq_epch) != 0:
            return
        try:
            # fetch all metrics. We use append (complexity amortized O(1)).
            for metric_name, metric_value in metrics.items():
                if metric_name != "epoch":
                    self.history[metric_name].append(metric_value)
                # Dealing with adding the epoch to the history.
                else:
                    # If the last value of the epoch is not the one we are currently trying to add:
                    if not len(self.history["epoch"]) or not self.history["epoch"][
                        -1
                    ] == (metric_value + 1):
                        # We shift by 1 for the plot.
                        self.history["epoch"].append(metric_value + 1)
            self.plot_history_prediction()
        # KeyError can happen when the metric is not in the history.
        except KeyError as e:
            raise AttributeError(
                f"KeyError found, potentially you have not instantiated "
                f"the logger with the key '{e.args[0]}'."
            )
        return

    def log_hyperparams(self, params, *args, **kwargs):
        self.hyper_params = params

    def _get_history_one_key(self, key):
        # Removes last point from validation scores.
        if key in self.history:
            # If contains a keyword indicating it is a validation metric.
            if any(
                val_keyword in key for val_keyword in TrainingHistoryLogger.val_keywords
            ):
                # Shifted list, the last validation loss corresponds to the model with parameters of the epoch n+1.
                return self.history[key][:-1]

            return self.history[key]

        else:
            raise KeyError(
                f"The key {key} does not exist in history. "
                f"If key is supposed to exist, has it been passed to the constructor of the logger?"
            )

    def fetch_score(self, metrics: typing.Iterable[str]):
        """
        Semantics:
            Gets the score if exists in the history. Removes last point from validation scores.

        Args:
            metrics (list<str>): the keys to fetch the result.

        Returns:
            list of lists of score.

        """
        # string or list of strings
        if isinstance(metrics, str):
            return [self._get_history_one_key(metrics)]
        else:
            res = [0] * len(metrics)
            for i, key in enumerate(metrics):
                res[i] = self._get_history_one_key(key)
            return res

    def plot_history_prediction(self):
        (epochs_loss,) = self.fetch_score(["epoch"])
        losses = self.fetch_score(self.metrics)
        len_loss = [len(lst) for lst in losses]
        if self.fig is not None and max(len_loss) == min(len_loss) == len(epochs_loss):
            self.ax.clear()

            # Ensure there's data to plot
            if len(epochs_loss) > 1:
                for i, (color, loss) in enumerate(zip(self.colors, losses)):
                    self.ax.plot(
                        epochs_loss,
                        loss,
                        color=color,
                        linestyle="-",
                        linewidth=2.5,
                        markersize=0.0,
                        label=self.metrics[i],
                    )

            self.ax.set_title("Dynamical Image of History Training")
            self.ax.set_xlabel("Epochs")
            self.ax.set_ylabel("Loss")
            self.ax.set_yscale("log")
            self.ax.legend(loc="best")

            plt.draw()
            plt.pause(0.001)

    @property
    def name(self):
        return "History_Dict_Logger"

    @property
    def version(self):
        return "V1.0"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass
