# Minimal TCN baseline for windowed pose tensors in PyTorch.
# Expects:
#   X: (N, T, F) float32
#   y: (N,) int64 class indices (0..C-1)
#
# Typical usage with your window tensors:
#   - If X is (N, T, K, C): reshape to (N, T, K*C)
#   - Map Tag strings to ints before training

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# TCN blocks
# ----------------------------

def _same_pad_1d(kernel_size: int, dilation: int) -> int:
    # padding that keeps length the same for stride=1
    return (kernel_size - 1) * dilation // 2


class ResidualTCNBlock(nn.Module):
    """
    A minimal residual dilated Conv1d block:
      Conv1d -> ReLU -> Dropout -> Conv1d -> ReLU -> Dropout + residual
    Non-causal "same-length" padding for simplicity.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        if kernel_size % 2 == 0:
            raise ValueError("Use an odd kernel_size (e.g. 3 or 5) to keep same-length padding simple.")
        super().__init__()
        pad = _same_pad_1d(kernel_size, dilation)

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)

        # Light init helps stability
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        res = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.drop(x)

        return x + res


class TCNBaseline(nn.Module):
    """
    Minimal TCN baseline:
      (B, T, F) -> (B, F, T)
      1x1 Conv to hidden channels
      Residual dilated blocks with dilation 1,2,4,... (or 1,2,4,8)
      Global average pool over time
      Linear head -> logits
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_channels: int = 128,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(in_features, hidden_channels, kernel_size=1)

        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** i
            blocks.append(ResidualTCNBlock(
                channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            ))
        self.tcn = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        returns logits: (B, num_classes)
        """
        x = x.transpose(1, 2)              # (B, F, T)
        x = self.input_proj(x)             # (B, H, T)
        x = F.relu(x)
        x = self.tcn(x)                    # (B, H, T)

        x = x.mean(dim=-1)                 # global average pool over time -> (B, H)
        logits = self.head(x)              # (B, C)
        return logits