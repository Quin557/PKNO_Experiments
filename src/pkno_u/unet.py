"""Small latent U-Nets and high-pass projection used by Stage3_1 PKNO-U."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class _Block1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Block2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentUNet1D(nn.Module):
    """One-level U-Net that preserves the latent channel count."""

    def __init__(self, channels: int, base_channels: int = 32) -> None:
        super().__init__()
        width = max(base_channels, channels)
        self.down = _Block1D(channels, width)
        self.middle = _Block1D(width, width)
        self.up = _Block1D(2 * width, width)
        self.out = nn.Conv1d(width, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.down(x)
        pooled = F.avg_pool1d(skip, kernel_size=2, ceil_mode=True)
        middle = self.middle(pooled)
        up = F.interpolate(middle, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.out(self.up(torch.cat([skip, up], dim=1)))


class LatentUNet2D(nn.Module):
    """One-level U-Net that preserves the latent channel count."""

    def __init__(self, channels: int, base_channels: int = 32) -> None:
        super().__init__()
        width = max(base_channels, channels)
        self.down = _Block2D(channels, width)
        self.middle = _Block2D(width, width)
        self.up = _Block2D(2 * width, width)
        self.out = nn.Conv2d(width, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.down(x)
        pooled = F.avg_pool2d(skip, kernel_size=2, ceil_mode=True)
        middle = self.middle(pooled)
        up = F.interpolate(middle, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.up(torch.cat([skip, up], dim=1)))


def high_pass(residual: torch.Tensor, cutoff: float = 0.5) -> torch.Tensor:
    """Keep only frequencies at or above a normalized radial cutoff."""

    if not 0.0 <= cutoff < 1.0:
        raise ValueError("cutoff must be in [0, 1).")
    if residual.ndim == 3:
        spectrum = torch.fft.rfft(residual)
        freq = torch.fft.rfftfreq(residual.shape[-1], device=residual.device, dtype=residual.dtype)
        threshold = cutoff * freq.max().clamp_min(1e-12)
        spectrum = spectrum * (freq >= threshold).reshape(1, 1, -1)
        return torch.fft.irfft(spectrum, n=residual.shape[-1])
    if residual.ndim == 4:
        spectrum = torch.fft.rfft2(residual)
        fx = torch.fft.fftfreq(residual.shape[-2], device=residual.device, dtype=residual.dtype)
        fy = torch.fft.rfftfreq(residual.shape[-1], device=residual.device, dtype=residual.dtype)
        grid_x, grid_y = torch.meshgrid(fx, fy, indexing="ij")
        radius = torch.sqrt(grid_x.square() + grid_y.square())
        threshold = cutoff * radius.max().clamp_min(1e-12)
        spectrum = spectrum * (radius >= threshold).reshape(1, 1, *radius.shape)
        return torch.fft.irfft2(spectrum, s=residual.shape[-2:])
    raise ValueError(f"Expected channels-first 1D or 2D tensor, got {tuple(residual.shape)}.")
