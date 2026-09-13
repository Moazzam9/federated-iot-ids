from __future__ import annotations

import torch
from torch import nn


class SmallMLP(nn.Module):
    """Small feed-forward neural network for binary intrusion detection."""

    def __init__(
        self,
        input_dim: int = 115,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return attack probabilities for a batch of samples."""
        return self.network(x)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
