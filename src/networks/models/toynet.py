import os
import sys

from src.networks.basic_nn import BasicNN
from src.networks.embeddings.time import TrigoTimeEmbedding

sys.path.append(os.getcwd() + "/Toy-Diffusion-Models")
import torch
import torch.nn as nn
import numpy as np


class ToyNet(nn.Module):
    # Model for diffusion where you pass the data (x) and the time step of the diffusion (t).
    def __init__(self, data_dim):
        super().__init__()
        self.time_embed_dim = 128
        dim = 256
        out_dim = data_dim

        self.trigotime_embed = TrigoTimeEmbedding(self.time_embed_dim)
        self.x_module = ResNet_FC(data_dim, dim, num_res_blocks=3)

        self.time_fcnn = BasicNN(
            self.time_embed_dim,
            [dim],
            dim,
            [True, True],
            [nn.SiLU()],
            0.0,
        )
        self.out_fcnn = BasicNN(
            dim,
            [dim],
            out_dim,
            [True, True],
            [nn.SiLU()],
            0.0,
        )
        self.out_module = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x, t: int or list[int]):
        t = timesteps_to_tensor(t, batch_size=x.shape[0]).to(x.device)

        t_emb = self.trigotime_embed(t)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
        out = self.out_module(x_out + t_out)
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


def timesteps_to_tensor(ts: int or list[int], batch_size):
    if isinstance(ts, list):
        assert (
            batch_size % len(ts) == 0
        ), "batch_size must be divisible by length of timesteps list"

    if isinstance(ts, int):
        return ts * torch.ones(batch_size)
    else:
        mini_batch_size = batch_size // len(ts)
        return torch.cat([ts[i] * torch.ones(mini_batch_size) for i in range(len(ts))])
