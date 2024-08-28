import torch
from pytorch_lightning import LightningDataModule
from sklearn.datasets import make_swiss_roll

from src.utils.fasttensordataloader import FastTensorDataLoader


class SwissRoll_Dataset(LightningDataModule):
    def __init__(self, data_size: int, use_2D_otherwise_3D: bool = True):
        super().__init__()

        self.use_2D_otherwise_3D = use_2D_otherwise_3D

        data, _ = make_swiss_roll(n_samples=data_size, noise=0.5)
        data = (data - data.mean(0)) / data.std(0)

        if use_2D_otherwise_3D:
            # If 2D, select only the first and third columns (x and z axes).
            data = data[:, [0, 2]]
            data_dim = 2
        else:
            # If 3D, use all three columns.
            data_dim = 3

        # Initialize the dataset with zeros
        train_data = torch.from_numpy(data).float().view(data_size, 1, data_dim)

        ### No this is wrong. Do not add zero here but after sequences created.
        # # Add zero beginning sequences.
        # train_data = torch.cat(
        #     (torch.zeros((data_size, 1, data_dim)), train_data), dim=1
        # )
        # # Add time dimension.
        # train_data = torch.cat(
        #     (train_data, torch.tensor([0.0]).repeat(data_size, 1).unsqueeze(-1)), dim=2
        # )

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

        train_data_np = self.train_in.numpy()
        val_data_np = self.val_in.numpy()

        plt.figure()

        if self.use_2D_otherwise_3D:
            # Plot for 2D data
            plt.scatter(
                train_data_np[:, 0, 0],
                train_data_np[:, 0, 1],
                c="b",
                label="Train Data",
            )
            plt.scatter(
                val_data_np[:, 0, 0],
                val_data_np[:, 0, 1],
                c="r",
                label="Validation Data",
            )
            plt.title("Train and Validation Data (2D)")
        else:

            # Plot for 3D data
            ax = plt.axes(projection="3d")
            ax.scatter(
                train_data_np[:, 0, 0],
                train_data_np[:, 0, 1],
                train_data_np[:, 0, 2],
                c="b",
                label="Train Data",
            )
            ax.scatter(
                val_data_np[:, 0, 0],
                val_data_np[:, 0, 1],
                val_data_np[:, 0, 2],
                c="r",
                label="Validation Data",
            )
            ax.set_title("Train and Validation Data (3D)")

        plt.legend()
        plt.show()
        return


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.set()

    mid_price_data_module = SwissRoll_Dataset(1000, use_2D_otherwise_3D=False)
    mid_price_data_module.plot_data()
    plt.show()
