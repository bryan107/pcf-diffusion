import torch
from pytorch_lightning import LightningDataModule
from torch.distributions import Normal, Categorical, MixtureSameFamily

from src.utils.fasttensordataloader import FastTensorDataLoader


class Mixture_Dataset(LightningDataModule):
    def __init__(self, data_size: int, len_ts: int):
        super().__init__()

        mix = Categorical(
            torch.ones(
                2,
            )
        )
        comp = Normal(torch.tensor([0 - 2.0, 0 + 2.0]), torch.tensor([0.5, 0.5]))
        data = MixtureSameFamily(mix, comp).sample([data_size, len_ts, 1])

        data = (data - data.mean()) / data.std()

        # Add zero beginning sequences.
        train_data = torch.cat((torch.zeros((data_size, 1, 1)), data), dim=1)

        # Add time dimension.
        train_data = torch.cat(
            (
                train_data,
                torch.linspace(0, 1, len_ts + 1).repeat(data_size, 1).unsqueeze(-1),
            ),
            dim=2,
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


if __name__ == "__main__":
    data = Mixture_Dataset(1000, 8)
    print(data.inputs.shape)
    print(data.train_in.shape)
    print(data.val_in.shape)
