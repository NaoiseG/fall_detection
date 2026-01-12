from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------
# Dataset
# ----------------------------

class WindowTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, T, F) or (N, T, K, C)
        y: (N,)
        """
        if X.ndim == 4:
            # (N, T, K, C) -> (N, T, K*C)
            N, T, K, C = X.shape
            X = X.reshape(N, T, K * C)
        assert X.ndim == 3, f"Expected X to be 3D (N,T,F). Got {X.shape}"

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]