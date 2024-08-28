import pickle

import torch
import torch.nn as nn


def save_obj(obj: object, filepath: str):
    """Generic function to save an object with different methods."""
    if filepath.endswith("pkl"):
        saver = pickle.dump
    elif filepath.endswith("pt"):
        saver = torch.save
    else:
        raise NotImplementedError()
    with open(filepath, "wb") as f:
        saver(obj, f)
    return 0


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        except:
            pass
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.kaiming_normal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        except:
            pass


# This looks very wrong. Perhaps dataloader.data or smthg
def loader_to_tensor(dl):
    tensor = []
    for x in dl:
        tensor.append(x[0])
    return torch.cat(tensor)


def loader_to_cond_tensor(dl):
    x_tensor = []
    y_tensor = []
    for x, y in dl:
        x_tensor.append(x)
        y_tensor.append(y)

    return torch.cat(x_tensor), torch.cat(y_tensor)


# TODO 28/08/2024 nie_k: Potentially not usable anymore, some changes made in the 4D one.
def cat_linspace_times(values_time_series):
    assert (
        len(values_time_series.shape) == 3
    ), f"Input shape must be [size, length, dim] but got {values_time_series.shape}"

    N, L, D = values_time_series.shape
    tt = (
        torch.linspace(0.0, 1, L, device=values_time_series.device)
        .view(1, -1, 1)
        .repeat(N, 1, 1)
    )
    return torch.cat([values_time_series, tt], dim=-1)


# TODO 12/08/2024 nie_k: Combine both methods for any shape >= 3
def cat_linspace_times_4D(values_time_series):
    assert (
        len(values_time_series.shape) == 4
    ), f"Input shape must be [size, length, dim] but got {values_time_series.shape}"

    S, N, L, D = values_time_series.shape
    tt = (
        torch.linspace(0.0, 1.0, L, device=values_time_series.device)
        .view(1, 1, -1, 1)
        .expand(S, N, L, 1)
    )
    return torch.cat([values_time_series, tt], dim=-1)
