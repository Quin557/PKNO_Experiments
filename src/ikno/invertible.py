"""Invertible observable dictionaries used by the IKNO baseline."""

from __future__ import annotations

import torch
from torch import nn


class ResidualMLP(nn.Module):
    """Three-linear-layer residual MLP used inside one coupling transform."""

    def __init__(self, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AdditiveCouplingBlock(nn.Module):
    """An exactly invertible additive coupling block with an implicit channel flip."""

    def __init__(self, channels: int, hidden_dim: int) -> None:
        super().__init__()
        if channels < 2 or channels % 2 != 0:
            raise ValueError("Additive coupling requires an even channel count of at least two.")
        self.channels = channels
        self.half_channels = channels // 2
        self.transform = ResidualMLP(self.half_channels, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[-1]}.")
        left, right = x.split(self.half_channels, dim=-1)
        updated_left = left + self.transform(right)
        return torch.cat([right, updated_left], dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.shape[-1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {y.shape[-1]}.")
        right, updated_left = y.split(self.half_channels, dim=-1)
        left = updated_left - self.transform(right)
        return torch.cat([left, right], dim=-1)


class InvertibleDictionary(nn.Module):
    """Pointwise IKNO observable map built from residual coupling blocks.

    The paper pads the two halves of the input time-delay vector separately
    before applying the invertible blocks.  This implementation uses a
    constant per-half width of ``observable_dim / 2`` across all blocks, the
    simplest valid instance of the paper's non-decreasing width schedule.
    """

    def __init__(
        self,
        input_channels: int,
        observable_dim: int,
        *,
        blocks: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if observable_dim < input_channels:
            raise ValueError("observable_dim must be at least input_channels.")
        if observable_dim % 2 != 0:
            raise ValueError("observable_dim must be even for additive coupling.")
        if blocks <= 0:
            raise ValueError("blocks must be positive.")
        self.input_channels = input_channels
        self.observable_dim = observable_dim
        self.left_input_channels = input_channels // 2
        self.right_input_channels = input_channels - self.left_input_channels
        self.half_observable_dim = observable_dim // 2
        if self.right_input_channels > self.half_observable_dim:
            raise ValueError("observable_dim is too small for the input split.")
        self.blocks = nn.ModuleList(
            [AdditiveCouplingBlock(observable_dim, hidden_dim) for _ in range(blocks)]
        )

    def _pad_split(self, x: torch.Tensor) -> torch.Tensor:
        left = x[..., : self.left_input_channels]
        right = x[..., self.left_input_channels :]
        left_padding = left.new_zeros(*left.shape[:-1], self.half_observable_dim - left.shape[-1])
        right_padding = right.new_zeros(*right.shape[:-1], self.half_observable_dim - right.shape[-1])
        return torch.cat([left, left_padding, right, right_padding], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.shape[-1]}.")
        z = self._pad_split(x)
        for block in self.blocks:
            z = block(z)
        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.observable_dim:
            raise ValueError(f"Expected {self.observable_dim} observable channels, got {z.shape[-1]}.")
        x = z
        for block in reversed(self.blocks):
            x = block.inverse(x)
        left = x[..., : self.half_observable_dim][..., : self.left_input_channels]
        right = x[..., self.half_observable_dim :][..., : self.right_input_channels]
        return torch.cat([left, right], dim=-1)
