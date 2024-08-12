import torch
from torch import nn

from src.utils.utils import init_weights


class DecodedLSTM(nn.Module):
    """
    A sequence-to-vector model based on LSTM.

    This model first projects the input sequence into a hidden space using a linear layer
    and an activation function. The projected sequence is then fed into an LSTM.
    Finally, the output of the LSTM is projected to the output space using another linear layer.

    Args:
        input_dim (int): Dimensionality of the input sequence.
        hidden_dim (int): Dimensionality of the hidden state.
        n_layers (int): Number of LSTM layers.
        out_dim (int): Dimensionality of the output.
        return_seq (bool): If True, returns the entire sequence of LSTM outputs. Otherwise,
                           returns only the last output.

    Returns:
        torch.Tensor: The output tensor.
    """

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
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.return_seq = return_seq

        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, out_dim)

        # Weight initialization
        self.apply(init_weights)

        self.activ_fn_1 = nn.LeakyReLU()
        self.activ_fn_2 = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 12/08/2024 nie_k:  missing hidden states in that lstm
        x = self.activ_fn_1(self.linear1(x))

        if self.return_seq:
            # Return entire sequence of LSTM outputs
            h = self.lstm(x)[0]
        else:
            # Return only the last output
            h = self.lstm(x)[0][:, -1:]

        x = self.linear(self.activ_fn_2(h))
        return x
