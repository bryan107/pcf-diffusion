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
