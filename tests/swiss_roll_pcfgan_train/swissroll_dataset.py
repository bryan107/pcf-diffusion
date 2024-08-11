import torch
from pytorch_lightning import LightningDataModule
from sklearn.datasets import make_swiss_roll

from src.utils.fasttensordataloader import FastTensorDataLoader


class SwissRoll_Dataset(LightningDataModule):
    def __init__(self, data_size: int):
        super().__init__()

        data, _ = make_swiss_roll(n_samples=data_size, noise=0.5)
        data = data[:, [0, 2]]

        # Initialize the dataset with zeros
        train_data = torch.from_numpy(data).float().view(data_size, 1, 2)

        # Add zero beginning sequences.
        train_data = torch.cat((torch.zeros((data_size, 1, 2)), train_data), dim=1)

        # Add time.
        train_data = torch.cat(
            (train_data, torch.tensor([0, 1]).repeat(data_size, 1).unsqueeze(-1)), dim=2
        )

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

        plt.scatter(
            train_data_np[:, 1, 0], train_data_np[:, 1, 1], c="b", label="Train Data"
        )
        plt.scatter(
            val_data_np[:, 1, 0], val_data_np[:, 1, 1], c="r", label="Validation Data"
        )

        plt.title("Train and Validation Data")
        plt.legend()
        plt.show()
        return


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.set()

    mid_price_data_module = SwissRoll_Dataset(1000)
    mid_price_data_module.plot_data()
    plt.show()
