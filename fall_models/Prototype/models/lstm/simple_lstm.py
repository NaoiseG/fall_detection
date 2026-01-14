from __future__ import annotations

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    Minimal LSTM baseline:
      (B, T, F)
        -> LSTM over time
        -> take last hidden state (or mean pool)
        -> Linear head -> logits (B, C)
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        pool: str = "last",  # "last" or "mean"
    ):
        super().__init__()

        if pool not in ("last", "mean"):
            raise ValueError("pool must be 'last' or 'mean'")

        self.pool = pool
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        # Note: dropout is only applied between LSTM layers when num_layers > 1
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,          # (B, T, F)
            bidirectional=bidirectional,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * self.num_directions),
            nn.Linear(hidden_size * self.num_directions, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        returns logits: (B, num_classes)
        """
        out, _ = self.lstm(x)  # out: (B, T, H * directions)

        if self.pool == "last":
            feat = out[:, -1, :]           # (B, H*dirs)
        else:  # "mean"
            feat = out.mean(dim=1)         # (B, H*dirs)

        logits = self.head(feat)           # (B, C)
        return logits