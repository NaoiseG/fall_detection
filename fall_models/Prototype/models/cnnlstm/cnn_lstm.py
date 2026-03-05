import torch
import torch.nn as nn
from typing import Optional


class CNNLSTMTwoHead(nn.Module):
    """
    Input:
      x: (B, T, F)

    Output:
      logits: (B, num_classes)

    If (num_keypoints, kp_channels) are provided AND F == K*Ck:
      uses a small Conv1d over joints (K) with channels=Ck per frame.

    Otherwise:
      falls back to a small Conv1d over the feature axis (length=F) per frame.
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_size: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        num_keypoints: Optional[int] = None,
        kp_channels: Optional[int] = None,
        pool: str = "last",          # "last" or "mean"
        kp_flatten_order: str = "KC" # "KC" assumes flatten from (K, Ck) -> K*Ck
    ):
        super().__init__()
        assert pool in ("last", "mean")
        assert kp_flatten_order in ("KC", "CK")

        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.hidden_size = int(hidden_size)
        self.lstm_layers = int(lstm_layers)
        self.dropout_p = float(dropout)

        self.num_keypoints = num_keypoints
        self.kp_channels = kp_channels
        self.pool = pool
        self.kp_flatten_order = kp_flatten_order

        # ----- Keypoint CNN (Option A1) -----
        # Only usable if kp_channels is known (Conv1d needs in_channels=Ck).
        if self.kp_channels is not None:
            Ck = int(self.kp_channels)
            self.kp_conv = nn.Sequential(
                nn.Conv1d(Ck, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_p),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_p),
            )
            self.kp_proj = nn.Sequential(
                nn.Linear(64, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_p),
            )
        else:
            self.kp_conv = None
            self.kp_proj = None

        # ----- Fallback CNN over feature axis (Option A2) -----
        # Treat each frame as a 1D signal of length F with 1 channel.
        self.feat_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(64, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
        )

        # ----- Temporal model -----
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout_p if self.lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        # Optional dropout before heads
        self.window_dropout = nn.Dropout(self.dropout_p)

        # ----- Classification head -----
        self.activity_head = nn.Linear(self.hidden_size, self.num_classes)

    def _encode_frames_keypoint_cnn(self, x_bt: torch.Tensor, K: int, Ck: int) -> torch.Tensor:
        """
        x_bt: (B*T, F) where F == K*Ck
        returns frame_emb: (B*T, E)
        """
        # Reshape flattened features into (B*T, K, Ck) or (B*T, Ck, K)
        if self.kp_flatten_order == "KC":
            x_kc = x_bt.view(-1, K, Ck)          # (B*T, K, Ck)
            x_ck = x_kc.permute(0, 2, 1).contiguous()  # (B*T, Ck, K)
        else:
            x_ck = x_bt.view(-1, Ck, K)          # (B*T, Ck, K)

        h = self.kp_conv(x_ck)                   # (B*T, 64, K)
        h = h.mean(dim=-1)                       # global avg pool over K -> (B*T, 64)
        frame_emb = self.kp_proj(h)              # (B*T, E)
        return frame_emb

    def _encode_frames_fallback(self, x_bt: torch.Tensor) -> torch.Tensor:
        """
        x_bt: (B*T, F)
        returns frame_emb: (B*T, E)
        """
        h = x_bt.unsqueeze(1)                    # (B*T, 1, F)
        h = self.feat_conv(h)                    # (B*T, 64, F)
        h = h.mean(dim=-1)                       # global avg pool over F -> (B*T, 64)
        frame_emb = self.feat_proj(h)            # (B*T, E)
        return frame_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)

        Returns:
          logits: (B, num_classes)
        """
        # Shape checks
        assert x.dim() == 3, f"Expected (B,T,F), got {tuple(x.shape)}"
        B, T, Fdim = x.shape

        # Flatten time into batch
        x_bt = x.reshape(B * T, Fdim)            # (B*T, F)

        # Choose encoder
        use_kp = (
            (self.num_keypoints is not None) and
            (self.kp_channels is not None) and
            (self.kp_conv is not None) and
            (Fdim == int(self.num_keypoints) * int(self.kp_channels))
        )

        if use_kp:
            K = int(self.num_keypoints)
            Ck = int(self.kp_channels)
            frame_emb = self._encode_frames_keypoint_cnn(x_bt, K, Ck)  # (B*T, E)
        else:
            frame_emb = self._encode_frames_fallback(x_bt)             # (B*T, E)

        # Restore time axis
        frame_emb = frame_emb.view(B, T, self.embed_dim)               # (B, T, E)

        # LSTM over time
        lstm_out, (h_n, c_n) = self.lstm(frame_emb)                    # lstm_out: (B, T, H)

        # Pool to window embedding
        if self.pool == "last":
            win_emb = lstm_out[:, -1, :]                               # (B, H)
        else:
            win_emb = lstm_out.mean(dim=1)                             # (B, H)

        win_emb = self.window_dropout(win_emb)

        # Head
        logits = self.activity_head(win_emb)                           # (B, C)
        return logits
