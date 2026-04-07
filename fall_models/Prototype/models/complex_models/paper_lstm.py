from __future__ import annotations

import torch
import torch.nn as nn


class _TemporalBatchNorm(nn.Module):
    """
    BatchNorm over the feature axis for sequence tensors shaped (B, T, C).
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(int(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B, T, C) input, got {tuple(x.shape)}")
        return self.bn(x.transpose(1, 2)).transpose(1, 2).contiguous()


class PaperLSTMClassifier(nn.Module):
    """
    Paper-inspired deep LSTM for pose-window classification.

    Approximation of the architecture described in the paper:
      - batch norm on the input sequence
      - 10 stacked LSTM layers
      - batch norm after each recurrent layer
      - final FC classifier

    Expected input:
      x: (B, T, F)
    Output:
      logits: (B, num_classes)
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_size: int = 80,
        num_layers: int = 10,
        dropout: float = 0.2,
        pool: str = "last",
    ):
        super().__init__()

        if pool not in {"last", "mean"}:
            raise ValueError("pool must be 'last' or 'mean'")
        if int(num_layers) < 1:
            raise ValueError("num_layers must be >= 1")

        self.pool = str(pool)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.input_bn = _TemporalBatchNorm(int(in_features))

        layers = []
        norms = []
        input_size = int(in_features)
        for _ in range(self.num_layers):
            layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=False,
                )
            )
            norms.append(_TemporalBatchNorm(self.hidden_size))
            input_size = self.hidden_size
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_size, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B, T, F) input, got {tuple(x.shape)}")

        x = self.input_bn(x)
        for lstm, norm in zip(self.layers, self.norms):
            x, _ = lstm(x)
            x = norm(x)
            x = self.dropout(x)

        if self.pool == "last":
            feat = x[:, -1, :]
        else:
            feat = x.mean(dim=1)

        return self.head(feat)
