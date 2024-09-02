import logging

import torch
from pytorch_lightning import LightningDataModule

logger = logging.getLogger(__name__)


from src.utils.fasttensordataloader import FastTensorDataLoader


class TrivialBM_Dataset(LightningDataModule):
    def __init__(self, data_size: int, batch_size: int):
        super().__init__()

        # Define the parameters
        SEQ_LEN = 1
        NUM_FEATURES = 1

        # Create the dataset with the specified shape
        # Last axis first dimension  DATA[0,:]: [gaussian random variable]

        # Initialize the dataset with zeros
        train_data = torch.zeros((data_size, SEQ_LEN, NUM_FEATURES))

        # Set the first dimension of the last axis
        train_data[:, :, 0] = torch.randn((data_size, 1))

        # # Set the second dimension of the last axis
        # train_data[:, :, 1] = torch.tensor([0, 1])

        self.inputs = train_data
        self.batch_size = batch_size

        training_size = int(80.0 / 100.0 * len(self.inputs))
        self.train_in = self.inputs[:training_size]
        self.val_in = self.inputs[training_size:]
        return

    def train_dataloader(self):
        return FastTensorDataLoader(self.train_in, batch_size=self.batch_size)

    def val_dataloader(self):
        return FastTensorDataLoader(self.val_in, batch_size=self.batch_size)

    def test_dataloader(self):
        return FastTensorDataLoader(self.inputs, batch_size=self.batch_size)

    def plot_data(self):
        import matplotlib.pyplot as plt

        print(self.inputs, "\n", self.inputs.shape)

        plt.figure()
        train_data_np = self.train_in.numpy()
        val_data_np = self.val_in.numpy()

        # Plot histograms with KDE
        sns.distplot(
            train_data_np[:, 0, 0],
            kde=True,
            color="blue",
            label="Train Data",
            hist=True,
        )

        sns.distplot(
            val_data_np[:, 0, 0],
            kde=True,
            color="red",
            label="Validation Data",
            hist=True,
        )
        plt.pause(0.1)
        plt.title("Histogram with KDE for Train and Validation Data")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        return


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.set()

    torch.manual_seed(0)

    mid_price_data_module = TrivialBM_Dataset(100_000, 1_000_000)
    mid_price_data_module.plot_data()
    plt.show()
