import torch
from torch import nn

from src.utils import init_weights


class DecodedLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        out_dim: int,
        return_seq=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.model = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_dim, out_dim)

        # Todo: could be just self.apply
        self.model.apply(init_weights)
        self.linear1.apply(init_weights)
        self.linear.apply(init_weights)
        self.return_seq = return_seq
        self.activ_fn_1 = nn.LeakyReLU()
        self.activ_fn_2 = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activ_fn(self.linear1(x))

        if self.return_seq:
            h = self.model(x)[0]
        else:
            h = self.model(x)[0][:, -1:]

        x = self.linear(self.activ_fn_2(h))
        return x
