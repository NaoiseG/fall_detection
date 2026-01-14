# models/stgcn/simple_stgcn.py
# Minimal ST-GCN-style baseline for windowed pose tensors in PyTorch.
#
# Expects:
#   X: (B, T, F) float32 where F = K * C (e.g. 17*3 = 51)
#   y: (B,) int64 class indices (0..C-1)
#
# This is intentionally simple:
#   - Spatial graph conv using a fixed normalized adjacency
#   - Temporal Conv2d over time dimension
#   - A few stacked ST-GCN blocks
#   - Global average pool over (T, K), linear head

from __future__ import annotations

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_adjacency(num_nodes: int, edges: List[Tuple[int, int]], self_loops: bool = True) -> torch.Tensor:
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if self_loops:
        A.fill_diagonal_(1.0)
    return A


def normalize_adjacency(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    deg = A.sum(dim=1)  # (K,)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


class SpatialGraphConv(nn.Module):
    """
    Spatial GCN over nodes for each time step.

    Input x: (B, Cin, T, K)
    Output:  (B, Cout, T, K)

    Operation:
      For each (B, Cin, T): apply adjacency mixing over K and then 1x1 conv over channels.
    """
    def __init__(self, cin: int, cout: int, A_norm: torch.Tensor):
        super().__init__()
        # register adjacency as buffer in parent and pass it in forward; keeping reference here for shape checks
        self.register_buffer("_A_norm_ref", A_norm)
        self.conv_1x1 = nn.Conv2d(cin, cout, kernel_size=(1, 1), bias=False)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, Cin, T, K), A_norm: (K, K)
        # mix nodes: einsum over K
        x = torch.einsum("bctk,kv->bctv", x, A_norm)  # (B, Cin, T, K)
        x = self.conv_1x1(x)                          # (B, Cout, T, K)
        return x


class STGCNBlock(nn.Module):
    """
    One ST-GCN block:
      Spatial graph conv -> BN/ReLU -> Temporal conv -> BN/ReLU -> Dropout (+ optional residual)
    """
    def __init__(
        self,
        cin: int,
        cout: int,
        A_norm: torch.Tensor,
        t_kernel: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        if t_kernel % 2 == 0:
            raise ValueError("Use an odd temporal kernel (e.g. 3, 5) for same-length padding")

        self.sgc = SpatialGraphConv(cin, cout, A_norm)

        self.bn1 = nn.BatchNorm2d(cout)
        self.tconv = nn.Conv2d(
            cout,
            cout,
            kernel_size=(t_kernel, 1),
            padding=((t_kernel - 1) // 2, 0),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(cout)
        self.drop = nn.Dropout(dropout)

        self.use_residual = use_residual
        if use_residual:
            if cin == cout:
                self.res = nn.Identity()
            else:
                self.res = nn.Conv2d(cin, cout, kernel_size=(1, 1), bias=False)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, Cin, T, K)
        res = self.res(x) if self.use_residual else 0

        x = self.sgc(x, A_norm)      # (B, Cout, T, K)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.tconv(x)            # (B, Cout, T, K)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.drop(x)
        return x + res if self.use_residual else x


class STGCNBaseline(nn.Module):
    """
    Minimal ST-GCN baseline for your windowed pose tensors.

    Input:
      x: (B, T, F) with F = K*C (e.g. 51)
    Internals:
      reshape -> (B, C, T, K)
      stack STGCN blocks
      global avg pool over (T, K)
      linear head
    """
    def __init__(
        self,
        num_nodes: int,
        node_features: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_blocks: int = 3,
        t_kernel: int = 3,
        dropout: float = 0.1,
        edges: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_features = node_features

        # Default edges for COCO-17-ish skeleton (0-indexed). Adjust if your kpt order differs.
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

        A = build_adjacency(num_nodes=num_nodes, edges=edges, self_loops=True)
        A_norm = normalize_adjacency(A)
        self.register_buffer("A_norm", A_norm)

        # Project input node features -> hidden channels (as a "channel" dim for conv2d)
        self.input_proj = nn.Conv2d(node_features, hidden_channels, kernel_size=(1, 1), bias=False)

        blocks = []
        cin = hidden_channels
        for _ in range(num_blocks):
            blocks.append(STGCNBlock(
                cin=cin,
                cout=hidden_channels,
                A_norm=A_norm,
                t_kernel=t_kernel,
                dropout=dropout,
                use_residual=True,
            ))
            cin = hidden_channels
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F), where F == K*C
        B, T, Fflat = x.shape
        expected = self.num_nodes * self.node_features
        if Fflat != expected:
            raise ValueError(f"Expected F={expected} (num_nodes*node_features), got F={Fflat}")

        # (B, T, K, C)
        x = x.view(B, T, self.num_nodes, self.node_features)
        # (B, C, T, K)
        x = x.permute(0, 3, 1, 2).contiguous()

        # input projection: (B, H, T, K)
        x = self.input_proj(x)

        for blk in self.blocks:
            x = blk(x, self.A_norm)

        # global average pool over time and nodes -> (B, H)
        x = x.mean(dim=(2, 3))
        logits = self.head(x)
        return logits