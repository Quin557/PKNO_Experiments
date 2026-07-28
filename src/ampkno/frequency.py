"""Conditioned AM-style frequency generators for AM-PKNO.

Stage4_0 combines two ideas already tested separately in this repository:

- PKNN/Stage3 keeps a parameter-independent shared dictionary and lets the
  Koopman operator vary with the condition.
- AM-FNO/Stage1 treats Fourier kernels as functions of frequency, using
  Chebyshev frequency bases and factorization to avoid one independent matrix
  per retained mode.

Here the generated object is a complex Koopman matrix, not an FNO kernel.
"""

from __future__ import annotations

import torch
from torch import nn

from amkno.frequency import ChebyshevFrequencyEmbedding


class ConditionedMatrixGenerator(nn.Module):
    """Generate complex matrices from frequency embeddings and conditions."""

    def __init__(
        self,
        *,
        freq_embed_dim: int,
        condition_embed_dim: int,
        observable_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
        output_scale: float = 0.02,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if freq_embed_dim <= 0 or condition_embed_dim <= 0 or observable_dim <= 0:
            raise ValueError("freq_embed_dim, condition_embed_dim, and observable_dim must be positive.")
        self.condition_embed_dim = condition_embed_dim
        self.observable_dim = observable_dim
        self.output_scale = output_scale

        layers: list[nn.Module] = []
        current = freq_embed_dim + condition_embed_dim
        for _ in range(depth):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(activation())
            current = hidden_dim
        layers.append(nn.Linear(current, 2 * observable_dim * observable_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, freq_embed: torch.Tensor, condition_embed: torch.Tensor) -> torch.Tensor:
        """Return ``[B, *freq_shape, O, O]`` complex Koopman matrices."""

        if condition_embed.ndim != 2 or condition_embed.shape[-1] != self.condition_embed_dim:
            raise ValueError(
                f"Expected condition_embed [B, {self.condition_embed_dim}], "
                f"got {tuple(condition_embed.shape)}."
            )
        freq_shape = freq_embed.shape[:-1]
        freq_flat = freq_embed.reshape(-1, freq_embed.shape[-1])
        batch = condition_embed.shape[0]
        n_freq = freq_flat.shape[0]

        freq_batch = freq_flat.unsqueeze(0).expand(batch, n_freq, freq_flat.shape[-1])
        cond_batch = condition_embed.unsqueeze(1).expand(batch, n_freq, condition_embed.shape[-1])
        raw = self.net(torch.cat([freq_batch, cond_batch], dim=-1))
        raw = raw.reshape(batch, *freq_shape, self.observable_dim, self.observable_dim, 2)
        return self.output_scale * torch.complex(raw[..., 0], raw[..., 1])


class ConditionedAxisFactorGenerator(nn.Module):
    """Generate one axis of a conditioned separable 2D Koopman matrix."""

    def __init__(
        self,
        *,
        freq_embed_dim: int,
        condition_embed_dim: int,
        observable_dim: int,
        rank: int = 1,
        hidden_dim: int = 128,
        depth: int = 2,
        output_scale: float = 0.02,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.condition_embed_dim = condition_embed_dim
        self.observable_dim = observable_dim
        self.rank = rank
        self.output_scale = output_scale

        layers: list[nn.Module] = []
        current = freq_embed_dim + condition_embed_dim
        for _ in range(depth):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(activation())
            current = hidden_dim
        layers.append(nn.Linear(current, 2 * observable_dim * observable_dim * rank))
        self.net = nn.Sequential(*layers)

    def forward(self, freq_embed: torch.Tensor, condition_embed: torch.Tensor) -> torch.Tensor:
        """Return ``[B, N, O, O, R]`` complex factors for one frequency axis."""

        if freq_embed.ndim != 2:
            raise ValueError(f"Axis freq_embed must be [N, F], got {tuple(freq_embed.shape)}.")
        if condition_embed.ndim != 2 or condition_embed.shape[-1] != self.condition_embed_dim:
            raise ValueError(
                f"Expected condition_embed [B, {self.condition_embed_dim}], "
                f"got {tuple(condition_embed.shape)}."
            )
        batch = condition_embed.shape[0]
        n_freq = freq_embed.shape[0]
        freq_batch = freq_embed.unsqueeze(0).expand(batch, n_freq, freq_embed.shape[-1])
        cond_batch = condition_embed.unsqueeze(1).expand(batch, n_freq, condition_embed.shape[-1])
        raw = self.net(torch.cat([freq_batch, cond_batch], dim=-1))
        raw = raw.reshape(batch, n_freq, self.observable_dim, self.observable_dim, self.rank, 2)
        return self.output_scale * torch.complex(raw[..., 0], raw[..., 1])


class ConditionedFactorizedMatrixGenerator2D(nn.Module):
    """Generate conditioned x/y factors for AM-PKNO's 2D Koopman update.

    The represented matrix is:

    ``K(kx, ky, c)[i, o] = sum_r Gx(kx, c)[i, o, r] * Gy(ky, c)[i, o, r]``.
    """

    def __init__(
        self,
        *,
        freq_embed_dim: int,
        condition_embed_dim: int,
        observable_dim: int,
        rank: int = 1,
        hidden_dim: int = 128,
        depth: int = 2,
        output_scale: float = 0.02,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        axis_scale = (output_scale / max(rank, 1)) ** 0.5
        self.x_generator = ConditionedAxisFactorGenerator(
            freq_embed_dim=freq_embed_dim,
            condition_embed_dim=condition_embed_dim,
            observable_dim=observable_dim,
            rank=rank,
            hidden_dim=hidden_dim,
            depth=depth,
            output_scale=axis_scale,
            activation=activation,
        )
        self.y_generator = ConditionedAxisFactorGenerator(
            freq_embed_dim=freq_embed_dim,
            condition_embed_dim=condition_embed_dim,
            observable_dim=observable_dim,
            rank=rank,
            hidden_dim=hidden_dim,
            depth=depth,
            output_scale=axis_scale,
            activation=activation,
        )

    def forward(
        self,
        freq_x_embed: torch.Tensor,
        freq_y_embed: torch.Tensor,
        condition_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.x_generator(freq_x_embed, condition_embed),
            self.y_generator(freq_y_embed, condition_embed),
        )


__all__ = [
    "ChebyshevFrequencyEmbedding",
    "ConditionedMatrixGenerator",
    "ConditionedFactorizedMatrixGenerator2D",
]
