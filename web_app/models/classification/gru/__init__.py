from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_adjacency(num_nodes: int, edges: List[Tuple[int, int]], self_loops: bool = True) -> torch.Tensor:
    """
    Build an undirected adjacency matrix A (num_nodes x num_nodes).
    """
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if self_loops:
        A.fill_diagonal_(1.0)
    return A


def normalize_adjacency(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Symmetric normalization: Â = D^{-1/2} A D^{-1/2}
    """
    deg = A.sum(dim=1)  # (K,)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


class GraphConv(nn.Module):
    """
    Simple GCN layer: X' = Â X W
    X: (B, K, Fin)
    """
    def __init__(self, fin: int, fout: int):
        super().__init__()
        self.lin = nn.Linear(fin, fout, bias=False)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # A_norm: (K, K)
        # x: (B, K, Fin)
        x = torch.matmul(A_norm, x)  # broadcasts: (K,K) @ (B,K,Fin) -> (B,K,Fin)
        x = self.lin(x)              # (B,K,Fout)
        return x


class GCNBaseline(nn.Module):
    """
    Minimal spatial GCN per frame + temporal average pooling.

    Pipeline:
      (B, T, F) -> reshape -> (B, T, K, C)
      per time-step: 2-layer GCN over K nodes
      pool nodes (mean) -> (B, T, H)
      pool time (mean)  -> (B, H)
      head -> logits (B, num_classes)
    """
    def __init__(
        self,
        num_nodes: int,
        node_features: int,
        num_classes: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
        edges: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.node_features = node_features
        self.hidden_size = hidden_size

        # Default: COCO-17-ish skeleton edges (0-indexed).
        # If your keypoint order differs, you should adjust these.
        if edges is None:
            edges = [
                (0, 1), (0, 2), (1, 3), (2, 4),        # head/eyes/ears-ish
                (5, 6),                                # shoulders
                (5, 7), (7, 9),                        # left arm
                (6, 8), (8, 10),                       # right arm
                (5, 11), (6, 12),                      # shoulders->hips
                (11, 12),                              # hips
                (11, 13), (13, 15),                    # left leg
                (12, 14), (14, 16),                    # right leg
            ]

        A = build_adjacency(num_nodes=num_nodes, edges=edges, self_loops=True)
        A_norm = normalize_adjacency(A)
        self.register_buffer("A_norm", A_norm)  # moves with .to(device)

        self.gcn1 = GraphConv(node_features, hidden_size)
        self.gcn2 = GraphConv(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F) where F == num_nodes * node_features
        """
        B, T, Fflat = x.shape
        expected = self.num_nodes * self.node_features
        if Fflat != expected:
            raise ValueError(f"Expected F={expected} (num_nodes*node_features), got F={Fflat}")

        # (B, T, K, C)
        x = x.view(B, T, self.num_nodes, self.node_features)

        # Apply GCN per time step
        # reshape to (B*T, K, C) for efficiency
        xt = x.reshape(B * T, self.num_nodes, self.node_features)

        h = self.gcn1(xt, self.A_norm)
        h = F.relu(h)
        h = self.drop(h)

        h = self.gcn2(h, self.A_norm)
        h = F.relu(h)
        h = self.drop(h)

        # Pool nodes: (B*T, H)
        h = h.mean(dim=1)

        # Back to (B, T, H)
        h = h.view(B, T, self.hidden_size)

        # Pool time: (B, H)
        h = h.mean(dim=1)

        # Classifier
        logits = self.head(h)  # (B, num_classes)
        return logits