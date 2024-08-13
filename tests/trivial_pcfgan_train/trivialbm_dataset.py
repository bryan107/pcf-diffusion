import logging

import torch
from pytorch_lightning import LightningDataModule

logger = logging.getLogger(__name__)


from src.utils.fasttensordataloader import FastTensorDataLoader


class TrivialBM_Dataset(LightningDataModule):
    def __init__(self, data_size: int):
        super().__init__()

        # Define the parameters
        SEQ_LEN = 2
        NUM_FEATURES = 2

        # Create the dataset with the specified shape
        # Last axis first dimension  DATA[0,:]: [0, gaussian random variable]
        # Last axis second dimension DATA[1,:]: [0, 1]

        # Initialize the dataset with zeros
        train_data = torch.zeros((data_size, SEQ_LEN, NUM_FEATURES))

        # Set the first dimension of the last axis
        train_data[:, :, 0] = torch.cat(
            (torch.zeros((data_size, 1)), torch.randn((data_size, 1))), dim=1
        )

        # Set the second dimension of the last axis
        train_data[:, :, 1] = torch.tensor([0, 1])

        self.inputs = train_data
        self.batch_size = 1_000_000

        training_size = int(90.0 / 100.0 * len(self.inputs))
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

        for seq in train_data_np:
            plt.plot(
                seq[:, 1],
                seq[:, 0],
                "b-x",
                label=(
                    "Train Data"
                    if "Train Data" not in plt.gca().get_legend_handles_labels()[1]
                    else ""
                ),
            )

        for seq in val_data_np:
            plt.plot(
                seq[:, 1],
                seq[:, 0],
                "r-o",
                label=(
                    "Validation Data"
                    if "Validation Data" not in plt.gca().get_legend_handles_labels()[1]
                    else ""
                ),
            )

        plt.title("Train and Validation Data")
        plt.xlabel("Feature 1 Time")
        plt.ylabel("Feature 0 Value")
        plt.legend()
        plt.show()
        return


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.set()

    mid_price_data_module = TrivialBM_Dataset(1000)
    mid_price_data_module.plot_data()
    plt.show()
