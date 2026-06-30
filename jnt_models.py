"""Models for the joint-angle SSL task. Two designs share one BiLSTM core:

  MaskedJointModel   (A): mask one of 3 channels, reconstruct it (all directions, one model).
  PairwiseJointModel (B): predict 1 target channel from the other 2 (one model per direction).

The core mirrors GraphTemporalRegressor (Gait_DP-RGNet/model/utils.py) so its weights can
later warm-start / constrain the main model's joint-angle head.
"""
import torch
import torch.nn as nn


class BiLSTMRegressor(nn.Module):
    """BiLSTM over the gait cycle + LayerNorm -> Linear -> ReLU -> Dropout -> Linear head."""

    def __init__(self, in_dim, out_dim, lstm_hidden=256, lstm_layers=2,
                 bidirectional=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Linear(lstm_out, lstm_out // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out // 2, out_dim),
        )

    def forward(self, x):                 # (B, T, in_dim) -> (B, T, out_dim)
        out, _ = self.lstm(x)
        return self.head(out)


class MaskedJointModel(nn.Module):
    """Design A. Input is [masked_angles(C), mask_flags(C)]; loss scored only on the
    masked channel. Output reconstructs all C channels."""

    def __init__(self, n_channels=3, lstm_hidden=256, lstm_layers=2,
                 bidirectional=True, dropout=0.3):
        super().__init__()
        self.n_channels = n_channels
        self.core = BiLSTMRegressor(
            in_dim=2 * n_channels, out_dim=n_channels,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
            bidirectional=bidirectional, dropout=dropout,
        )

    def forward(self, x, mask):
        """x: (B, T, C) angles. mask: (B, C) bool, True = channel to predict."""
        B, T, C = x.shape
        m = mask.unsqueeze(1).expand(B, T, C).float()
        feat = torch.cat([x * (1.0 - m), m], dim=-1)   # zero masked channels, append flags
        return self.core(feat)                         # (B, T, C)


class PairwiseJointModel(nn.Module):
    """Design B. One directional regressor: 2 source channels -> 1 target channel."""

    def __init__(self, in_dim=2, out_dim=1, lstm_hidden=256, lstm_layers=2,
                 bidirectional=True, dropout=0.3):
        super().__init__()
        self.core = BiLSTMRegressor(
            in_dim=in_dim, out_dim=out_dim,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
            bidirectional=bidirectional, dropout=dropout,
        )

    def forward(self, x):                 # (B, T, 2) -> (B, T, 1)
        return self.core(x)
