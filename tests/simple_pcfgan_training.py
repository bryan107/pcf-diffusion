import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.PCFGAN import PCFGANTrainer
from src.evaluations.test_metrics import get_standard_test_metrics
from src.networks.generators import LSTMGenerator
from src.utils import save_obj

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Define the parameters
data_size = 1000
seq_len = 2
time_series_dim = 2

# Create the dataset with the specified shape
# Last axis first dimension: 0, gaussian random variable
# Last axis second dimension: 0, 1

# Initialize the dataset with zeros
train_data = torch.zeros((data_size, seq_len, time_series_dim))

# Set the first dimension of the last axis
train_data[:, :, 0] = torch.cat(
    (torch.zeros((data_size, 1)), torch.randn((data_size, 1))), dim=1
)

# Set the second dimension of the last axis
train_data[:, :, 1] = torch.tensor([0, 1])

print(train_data, "\n", train_data.shape)

# plot data:
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

plt.figure()
train_data_np = train_data.numpy()
for seq in train_data_np:
    plt.plot(seq[:, 1], seq[:, 0], "b-x")
plt.pause(0.1)

train_data = DataLoader(
    [train_data],
    batch_size=data_size,
    shuffle=True,
)


class Config:
    """Adapter to convert a dictionary to an object with properties/fields."""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


config = {
    "device": "cuda",
    "add_time": False,
    "lr_G": 0.001,
    "lr_D": 0.001,
    "D_steps_per_G_step": 2,
    # NUM EPOCHS
    "num_epochs": 1001,
    "G_input_dim": 2,
    "G_hidden_dim": 8,
    "input_dim": 2,
    "M_num_samples": 6,
    "M_hidden_dim": 10,
    "lr_M": 0.005,
    "Lambda1": 50,
    "Lambda2": 1,
    "gamma": 0.97,
    # WIP NUM ELEMENT IN SEQ?
    "n_lags": 2,
    "batch_size": data_size,
    "exp_dir": "./results/",
    "gan_algo": "PCFGAN",
    "swa_step_start": 25000,
}
config = Config(config)

trainer = PCFGANTrainer(
    generator=LSTMGenerator(
        input_dim=2,
        hidden_dim=8,
        output_dim=2,
        n_layers=1,
        noise_scale=1.0,
        BM=True,
        activation=nn.Identity(),
    ),
    train_dataset=train_data,
    config=config,
    # wip: THESE TWO SEEM UNUSED??
    test_metrics_train=get_standard_test_metrics(train_data),
    test_metrics_test=get_standard_test_metrics(train_data),
    # WIP I THINK BATCH SIZE DOES SMTHG DIFFERENT
)

trainer.fit(config.device)
save_obj(
    trainer.generator.state_dict(),
    os.path.join(config.exp_dir, "generator_state_dict.pt"),
)
