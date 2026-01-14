from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    Minimal MLP over flattened windows.
    """
    def __init__(
        self,
        T: int,
        in_features: int,
        num_classes: int,
        hidden_sizes=(256, 128),
        dropout: float = 0.2,
    ):
        super().__init__()

        if T <= 0:
            raise ValueError("T must be a positive integer")

        input_dim = T * in_features

        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        returns logits: (B, num_classes)
        """
        x = x.reshape(x.size(0), -1)  # (B, T*F)
        return self.net(x)