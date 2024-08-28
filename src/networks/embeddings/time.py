import math

import numpy as np
import torch
import torch.nn as nn


class TrigoTimeEmbedding(nn.Module):
    def __init__(self, embed_size, min_time: float = 0.0, max_time: float = 1.0):
        # Min and max time for scaling the time embeddings. If you want it to have no impact, set it to 0 and 1.
        super().__init__()
        assert (
            embed_size % 2 == 0
        ), f"Embedding size must be even but got {embed_size:d}."
        assert (
            embed_size > 0
        ), f"Embedding size must be positive but got {embed_size:d}."
        assert (
            min_time < max_time
        ), f"Min time must be less than max time but got {min_time:.2f} and {max_time:.2f}."

        self.embed_size = embed_size
        self.min_time = min_time
        self.max_time = max_time

        self.learnable_weights = False

        if self.learnable_weights:
            self.embedding_weights = nn.Linear(1, embed_size // 2, bias=False)
            torch.nn.init.xavier_uniform_(
                self.embedding_weights.weight, gain=nn.init.calculate_gain("tanh")
            )
        else:
            # Parameter to be moved on device.
            self.embedding_weights = nn.Parameter(
                self._create_embedding_weights(embed_size), requires_grad=False
            )

    def forward(self, times):
        # phi shape: (batch_size, Linput, embed_size // 2)
        times = (times - self.min_time) / (
            self.max_time - self.min_time
        )  # normalise the time to be between 0 and 1
        if self.learnable_weights:
            phi = self.embedding_weights(times)
        else:
            phi = torch.matmul(times, self.embedding_weights)
        pe_sin = torch.sin(phi * 2.0 * math.pi)
        pe_cos = torch.cos(phi * 2.0 * math.pi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        # Scaling to ensure that the magnitude of the embeddings does not change with the embedd size, as Transformers.
        return pe / math.sqrt(self.embed_size)

    @staticmethod
    def _create_embedding_weights(d_model):
        return (
            torch.from_numpy(
                np.array(
                    [
                        TrigoTimeEmbedding._get_frequency_time_embed(j, d_model)
                        for j in range(0, d_model, 2)
                    ]
                )
            ).float()
            # Reshaping to be a matrix that increases number of dimensions.
            .reshape(1, -1)
        )

    @staticmethod
    def _get_frequency_time_embed(j, d_model):
        return 10 ** (2.0 * j / d_model)
