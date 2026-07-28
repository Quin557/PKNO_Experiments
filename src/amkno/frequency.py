"""Frequency embeddings and amortized complex matrix generation."""

from __future__ import annotations

import torch
from torch import nn


class ChebyshevFrequencyEmbedding(nn.Module):
    """Embed normalized FFT frequencies with Chebyshev basis functions.

    AM-FNO's MLP variant uses orthogonal basis functions before the MLP because
    a vanilla MLP tends to underfit oscillatory frequency-to-kernel maps.  Here
    the same idea is used to generate KNO's frequency-wise Koopman matrices.
    """

    def __init__(self, spatial_dim: int, basis_dim: int = 32, include_raw: bool = True) -> None:
        super().__init__()
        if spatial_dim <= 0 or basis_dim <= 0:
            raise ValueError("spatial_dim and basis_dim must be positive.")
        self.spatial_dim = spatial_dim
        self.basis_dim = basis_dim
        self.include_raw = include_raw
        degrees = torch.arange(1, basis_dim + 1, dtype=torch.float32)
        self.register_buffer("degrees", degrees, persistent=False)
        self.output_dim = spatial_dim * basis_dim + (spatial_dim if include_raw else 0)

    def forward(self, freq: torch.Tensor) -> torch.Tensor:
        if freq.shape[-1] != self.spatial_dim:
            raise ValueError(f"Expected freq dim {self.spatial_dim}, got {freq.shape[-1]}.")
        # Chebyshev T_n(x) is defined on [-1, 1]. FFT frequencies already live
        # inside that range after normalization by torch.fft.fftfreq/rfftfreq.
        freq_clamped = freq.clamp(-1.0, 1.0)
        theta = torch.acos(freq_clamped)
        basis = torch.cos(theta.unsqueeze(-1) * self.degrees)
        basis = basis.reshape(*freq.shape[:-1], self.spatial_dim * self.basis_dim)
        if not self.include_raw:
            return basis
        return torch.cat([freq, basis], dim=-1)


class AmortizedMatrixGenerator(nn.Module):
    """Generate complex Koopman matrices from frequency and optional condition."""

    def __init__(
        self,
        freq_embed_dim: int,
        observable_dim: int,
        condition_dim: int = 0,
        hidden_dim: int = 128,
        depth: int = 2,
        output_scale: float = 0.05,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if freq_embed_dim <= 0 or observable_dim <= 0:
            raise ValueError("freq_embed_dim and observable_dim must be positive.")
        if condition_dim < 0:
            raise ValueError("condition_dim cannot be negative.")
        self.observable_dim = observable_dim
        self.condition_dim = condition_dim
        self.output_scale = output_scale

        layers: list[nn.Module] = []
        current = freq_embed_dim + condition_dim
        for _ in range(depth):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(activation())
            current = hidden_dim
        layers.append(nn.Linear(current, 2 * observable_dim * observable_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, freq_embed: torch.Tensor, condition_embed: torch.Tensor | None = None) -> torch.Tensor:
        freq_shape = freq_embed.shape[:-1]
        freq_flat = freq_embed.reshape(-1, freq_embed.shape[-1])
        n_freq = freq_flat.shape[0]

        if self.condition_dim == 0:
            raw = self.net(freq_flat)
            raw = raw.reshape(*freq_shape, self.observable_dim, self.observable_dim, 2)
            return self.output_scale * torch.complex(raw[..., 0], raw[..., 1])

        if condition_embed is None:
            raise ValueError("condition_embed is required when condition_dim > 0.")
        if condition_embed.ndim != 2 or condition_embed.shape[-1] != self.condition_dim:
            raise ValueError(
                f"Expected condition_embed [B, {self.condition_dim}], got {tuple(condition_embed.shape)}."
            )
        batch = condition_embed.shape[0]
        freq_batch = freq_flat.unsqueeze(0).expand(batch, n_freq, freq_flat.shape[-1])
        cond_batch = condition_embed.unsqueeze(1).expand(batch, n_freq, condition_embed.shape[-1])
        raw = self.net(torch.cat([freq_batch, cond_batch], dim=-1))
        raw = raw.reshape(batch, *freq_shape, self.observable_dim, self.observable_dim, 2)
        return self.output_scale * torch.complex(raw[..., 0], raw[..., 1])
