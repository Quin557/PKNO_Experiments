"""Lightweight high-frequency residual path (one convolutional block)."""

from __future__ import annotations

import torch
from torch import nn


class HighFrequencyResidual(nn.Module):
    def __init__(self, spatial_dim: int, input_channels: int, output_channels: int, hidden: int = 16) -> None:
        super().__init__()
        conv = nn.Conv1d if spatial_dim == 1 else nn.Conv2d
        self.net = nn.Sequential(conv(input_channels, hidden, 3, padding=1), nn.GELU(), conv(hidden, output_channels, 3, padding=1))
        self.raw_scale = nn.Parameter(torch.tensor(0.10033))  # 0.1*tanh(.) ~= 0.01

    def forward(self, latest: torch.Tensor) -> torch.Tensor:
        channels_first = latest.permute(0, 2, 1) if latest.ndim == 3 else latest.permute(0, 3, 1, 2)
        residual = self.net(channels_first)
        residual = residual.permute(0, 2, 1) if latest.ndim == 3 else residual.permute(0, 2, 3, 1)
        return 0.1 * torch.tanh(self.raw_scale) * residual
