import logging

import torch
from pytorch_lightning import LightningDataModule

logger = logging.getLogger(__name__)


from src.utils.fasttensordataloader import FastTensorDataLoader


class TrivialBM_Dataset(LightningDataModule):
    def __init__(self, data_size: int, batch_size: int, bimodal: int = True):
        super().__init__()

        # Define the parameters
        SEQ_LEN = 1
        NUM_FEATURES = 1

        if not bimodal:
            train_data = torch.randn((data_size, SEQ_LEN, NUM_FEATURES))

        else:
            # Parameters for the two different Gaussian distributions
            mean1, std1 = 0.0, 1.0  # Mean and standard deviation for the first Gaussian
            mean2, std2 = 10.0, 1.0  # Mean and standard deviation for the second Gaussian

            # Create a binary mask to select between the two Gaussians
            mask = (
                torch.rand(data_size) < 0.3
            )  # Randomly selects True or False for each sample

            # Generate data from the first Gaussian
            data1 = torch.randn(data_size, SEQ_LEN, NUM_FEATURES) * std1 + mean1

            # Generate data from the second Gaussian
            data2 = torch.randn(data_size, SEQ_LEN, NUM_FEATURES) * std2 + mean2

            # Combine the two datasets using the mask
            train_data = torch.where(mask.view(-1, 1, 1), data1, data2)

        # Standardize the data:
        mean = train_data.mean()
        std = train_data.std()
        train_data = (train_data - mean) / std

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
