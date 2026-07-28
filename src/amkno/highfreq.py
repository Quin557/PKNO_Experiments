"""Optional output-side high-frequency residual branches for AM-KNO."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HighFrequencyResidual1D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, output_channels, 3, padding=1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        latest = history[..., -1:].permute(0, 2, 1)
        high = latest - F.avg_pool1d(latest, kernel_size=5, stride=1, padding=2)
        return self.net(high).permute(0, 2, 1)


class HighFrequencyResidual2D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_channels, 3, padding=1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        latest = history[..., -1:].permute(0, 3, 1, 2)
        high = latest - F.avg_pool2d(latest, kernel_size=5, stride=1, padding=2)
        return self.net(high).permute(0, 2, 3, 1)
