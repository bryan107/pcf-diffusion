import logging

import numpy as np
import torch
import torch.nn as nn

from src.networks.basic_nn import BasicNN
from src.networks.embeddings.time import TrigoTimeEmbedding

logger = logging.getLogger(__name__)


class ToyNet(nn.Module):
    # Model for diffusion where you pass the data (x) and the time step of the diffusion (t).
    def __init__(self, data_dim):
        super().__init__()
        self.input_dim = data_dim
        self.hidden_dim = 512
        self.time_embed_dim = 64
        self.output_dim = data_dim

        self.trigotime_embed = TrigoTimeEmbedding(self.time_embed_dim, 0, 1)

        self.data_resnet = BasicNN(
            self.input_dim,
            [self.hidden_dim],
            self.hidden_dim,
            [True, True],
            [nn.SiLU()],
            0.0,
        )

        # Transforms time embeddings
        self.time_fcnn = BasicNN(
            self.time_embed_dim,
            [self.hidden_dim],
            self.hidden_dim,
            [True, True],
            [nn.SiLU()],
            0.0,
        )
        self.out_fcnn = BasicNN(
            self.hidden_dim,
            [self.hidden_dim, self.hidden_dim, self.hidden_dim],
            self.output_dim,
            [True, True, True, True],
            [nn.SiLU(), nn.SiLU(), nn.SiLU()],
            0.0,
        )
        return

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Be careful, t needs to be a scalar (0D or 1D in torch).
        t_emb = self.trigotime_embed(t.view(1))
        t_out = self.time_fcnn(t_emb)
        x_out = self.data_resnet(x)
        out = self.out_fcnn(x_out + t_out)
        return out


# --------------------  Utils for Toy model  --------------------#
# WIP: unify with Hang's ResNet
class ResNet_FC(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_res_blocks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.map = nn.Linear(data_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [self.build_res_block() for _ in range(num_res_blocks)]
        )

    def build_linear(self, in_features, out_features):
        linear = nn.Linear(in_features, out_features)
        return linear

    def build_res_block(self):
        hid = self.hidden_dim
        layers = []
        widths = [hid] * 4
        for i in range(len(widths) - 1):
            layers.append(self.build_linear(widths[i], widths[i + 1]))
            layers.append(nn.SiLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.map(x)
        for res_block in self.res_blocks:
            h = (h + res_block(h)) / np.sqrt(2)
        return h
