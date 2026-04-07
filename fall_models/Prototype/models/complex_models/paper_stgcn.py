from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.stgcn.simple_stgcn import STGCNBlock, build_adjacency, normalize_adjacency


class PaperSTGCNClassifier(nn.Module):
    """
    Deeper paper-inspired ST-GCN for windowed pose tensors.

    Design choices track the paper description as closely as the current
    pipeline allows:
      - 10 ST-GCN blocks
      - temporal kernel size 9
      - channel progression 64 -> 128 -> 256
      - residual connections and dropout

    Expected input:
      x: (B, T, K*C) where K=num_nodes
    Output:
      logits: (B, num_classes)
    """

    def __init__(
        self,
        num_nodes: int,
        node_features: int,
        num_classes: int,
        channels: Sequence[int] = (64, 64, 64, 64, 128, 128, 128, 256, 256, 256),
        t_kernel: int = 9,
        dropout: float = 0.2,
        edges: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        if not channels:
            raise ValueError("channels must be non-empty")
        if int(t_kernel) % 2 == 0:
            raise ValueError("t_kernel must be odd")

        self.num_nodes = int(num_nodes)
        self.node_features = int(node_features)
        channel_list = [int(c) for c in channels]

        if edges is None:
            edges = [
                (0, 1), (0, 2), (1, 3), (2, 4),
                (5, 6),
                (5, 7), (7, 9),
                (6, 8), (8, 10),
                (5, 11), (6, 12),
                (11, 12),
                (11, 13), (13, 15),
                (12, 14), (14, 16),
            ]

        A = build_adjacency(num_nodes=self.num_nodes, edges=edges, self_loops=True)
        self.register_buffer("A_norm", normalize_adjacency(A))

        self.data_bn = nn.BatchNorm1d(self.num_nodes * self.node_features)
        self.input_proj = nn.Conv2d(self.node_features, channel_list[0], kernel_size=(1, 1), bias=False)

        blocks = []
        cin = channel_list[0]
        for cout in channel_list:
            blocks.append(
                STGCNBlock(
                    cin=cin,
                    cout=cout,
                    A_norm=self.A_norm,
                    t_kernel=int(t_kernel),
                    dropout=float(dropout),
                    use_residual=True,
                )
            )
            cin = cout
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(channel_list[-1]),
            nn.Dropout(float(dropout)),
            nn.Linear(channel_list[-1], int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B, T, F) input, got {tuple(x.shape)}")

        batch_size, time_steps, flat_features = x.shape
        expected = self.num_nodes * self.node_features
        if int(flat_features) != int(expected):
            raise ValueError(f"Expected F={expected} (num_nodes*node_features), got F={flat_features}")

        x = x.view(batch_size, time_steps, self.num_nodes, self.node_features)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, self.num_nodes * self.node_features, time_steps)
        x = self.data_bn(x)
        x = x.view(batch_size, self.num_nodes, self.node_features, time_steps)
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, self.A_norm)

        x = x.mean(dim=(2, 3))
        return self.head(x)
