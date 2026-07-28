"""PyTorch parameter-conditioned Koopman operators for Stage 3.

This file ports the PKNN idea ``K(u) = NN_K(u)`` into KNO's Fourier-domain
operator.  No TensorFlow/Keras code is imported: ``nn.Module.forward`` replaces
``Layer.call`` and ``torch.einsum`` replaces ``tf.einsum``.
"""

from __future__ import annotations

import torch
from torch import nn


class ConditionEncoder(nn.Module):
    """Encode explicit physical conditions and current-state summaries."""

    def __init__(
        self,
        condition_dim: int,
        state_embed_dim: int,
        output_dim: int = 128,
        hidden_dim: int = 128,
        depth: int = 2,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if condition_dim < 0 or state_embed_dim < 0:
            raise ValueError("condition_dim and state_embed_dim cannot be negative.")
        in_dim = condition_dim + state_embed_dim
        if in_dim <= 0:
            raise ValueError("At least one condition input is required.")

        layers: list[nn.Module] = []
        current = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(activation())
            current = hidden_dim
        layers.append(nn.Linear(current, output_dim))
        layers.append(activation())
        self.net = nn.Sequential(*layers)
        self.condition_dim = condition_dim
        self.state_embed_dim = state_embed_dim
        self.output_dim = output_dim

    def forward(self, condition: torch.Tensor, state_embed: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 2:
            raise ValueError(f"condition must be [B, C], got {tuple(condition.shape)}.")
        if state_embed.ndim != 2:
            raise ValueError(f"state_embed must be [B, E], got {tuple(state_embed.shape)}.")
        if condition.shape[0] != state_embed.shape[0]:
            raise ValueError("condition and state_embed batch sizes differ.")
        if condition.shape[-1] != self.condition_dim:
            raise ValueError(f"Expected condition dim {self.condition_dim}, got {condition.shape[-1]}.")
        if state_embed.shape[-1] != self.state_embed_dim:
            raise ValueError(f"Expected state embed dim {self.state_embed_dim}, got {state_embed.shape[-1]}.")
        return self.net(torch.cat([condition, state_embed], dim=-1))


class KoopmanMatrixGenerator(nn.Module):
    """Generate complex Koopman matrix corrections from ``freq`` and ``condition``."""

    def __init__(
        self,
        freq_dim: int,
        condition_embed_dim: int,
        observable_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
        delta_scale: float = 0.05,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if freq_dim <= 0 or condition_embed_dim <= 0 or observable_dim <= 0:
            raise ValueError("freq_dim, condition_embed_dim, and observable_dim must be positive.")
        self.observable_dim = observable_dim
        self.delta_scale = delta_scale

        layers: list[nn.Module] = []
        current = freq_dim + condition_embed_dim
        for _ in range(depth):
            layers.append(nn.Linear(current, hidden_dim))
            layers.append(activation())
            current = hidden_dim
        layers.append(nn.Linear(current, 2 * observable_dim * observable_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, freq: torch.Tensor, condition_embed: torch.Tensor) -> torch.Tensor:
        """Return a complex correction with shape ``[B, *freq_shape, O, O]``."""

        if freq.ndim < 2:
            raise ValueError(f"freq must be [*freq_shape, freq_dim], got {tuple(freq.shape)}.")
        if condition_embed.ndim != 2:
            raise ValueError(
                f"condition_embed must be [B, condition_embed_dim], got {tuple(condition_embed.shape)}."
            )

        batch = condition_embed.shape[0]
        freq_shape = freq.shape[:-1]
        freq_flat = freq.reshape(-1, freq.shape[-1])
        n_freq = freq_flat.shape[0]

        freq_batch = freq_flat.unsqueeze(0).expand(batch, n_freq, freq_flat.shape[-1])
        cond_batch = condition_embed.unsqueeze(1).expand(batch, n_freq, condition_embed.shape[-1])
        raw = self.net(torch.cat([freq_batch, cond_batch], dim=-1))
        raw = raw.reshape(batch, n_freq, self.observable_dim, self.observable_dim, 2)
        delta = torch.complex(raw[..., 0], raw[..., 1])
        return self.delta_scale * delta.reshape(batch, *freq_shape, self.observable_dim, self.observable_dim)


class ParameterizedKoopmanOperator1D(nn.Module):
    """Frequency-wise ``K_k = K0_k + DeltaK(k, c, u_embed)`` for 1D KNO."""

    def __init__(
        self,
        observable_dim: int,
        modes: int = 16,
        condition_embed_dim: int = 128,
        generator_hidden_dim: int = 128,
        generator_depth: int = 2,
        delta_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.observable_dim = observable_dim
        self.modes = modes
        scale = 1.0 / (observable_dim * observable_dim)
        self.base_matrix = nn.Parameter(
            scale * torch.randn(modes, observable_dim, observable_dim, dtype=torch.cfloat)
        )
        self.generator = KoopmanMatrixGenerator(
            freq_dim=1,
            condition_embed_dim=condition_embed_dim,
            observable_dim=observable_dim,
            hidden_dim=generator_hidden_dim,
            depth=generator_depth,
            delta_scale=delta_scale,
        )

    @staticmethod
    def _time_march(x_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # x_ft: [B, O_in, M], weights: [B, M, O_in, O_out].
        return torch.einsum("bim,bmio->bom", x_ft, weights)

    def _freq(self, modes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        denom = max(modes - 1, 1)
        return (torch.arange(modes, device=device, dtype=dtype) / denom).unsqueeze(-1)

    def make_weights(self, x: torch.Tensor, condition_embed: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"1D operator expects [B, O, X], got {tuple(x.shape)}.")
        x_ft = torch.fft.rfft(x)
        modes = min(self.modes, x_ft.shape[-1])
        freq = self._freq(modes, x.device, x.real.dtype)
        delta = self.generator(freq, condition_embed)
        return self.base_matrix[:modes].unsqueeze(0) + delta

    def forward(
        self,
        x: torch.Tensor,
        condition_embed: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"1D operator expects [B, O, X], got {tuple(x.shape)}.")
        x_ft = torch.fft.rfft(x)
        if weights is None:
            weights = self.make_weights(x, condition_embed)
        modes = weights.shape[1]
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :modes] = self._time_march(x_ft[:, :, :modes], weights)
        return torch.fft.irfft(out_ft, n=x.shape[-1])


class ParameterizedKoopmanOperator2D(nn.Module):
    """Frequency-wise ``K_k = K0_k + DeltaK(k, c, u_embed)`` for 2D KNO."""

    def __init__(
        self,
        observable_dim: int,
        modes_x: int = 16,
        modes_y: int = 16,
        condition_embed_dim: int = 128,
        generator_hidden_dim: int = 128,
        generator_depth: int = 2,
        delta_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.observable_dim = observable_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (observable_dim * observable_dim)
        self.base_matrix = nn.Parameter(
            scale * torch.randn(modes_x, modes_y, observable_dim, observable_dim, dtype=torch.cfloat)
        )
        self.generator = KoopmanMatrixGenerator(
            freq_dim=2,
            condition_embed_dim=condition_embed_dim,
            observable_dim=observable_dim,
            hidden_dim=generator_hidden_dim,
            depth=generator_depth,
            delta_scale=delta_scale,
        )

    @staticmethod
    def _time_march(x_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # x_ft: [B, O_in, MX, MY], weights: [B, MX, MY, O_in, O_out].
        return torch.einsum("bixy,bxyio->boxy", x_ft, weights)

    def _freq(self, modes_x: int, modes_y: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        denom_x = max(modes_x - 1, 1)
        denom_y = max(modes_y - 1, 1)
        kx = torch.arange(modes_x, device=device, dtype=dtype) / denom_x
        ky = torch.arange(modes_y, device=device, dtype=dtype) / denom_y
        grid_x, grid_y = torch.meshgrid(kx, ky, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1)

    def make_weights(self, x: torch.Tensor, condition_embed: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"2D operator expects [B, O, X, Y], got {tuple(x.shape)}.")
        x_ft = torch.fft.rfft2(x)
        modes_x = min(self.modes_x, x_ft.shape[-2])
        modes_y = min(self.modes_y, x_ft.shape[-1])
        freq = self._freq(modes_x, modes_y, x.device, x.real.dtype)
        delta = self.generator(freq, condition_embed)
        return self.base_matrix[:modes_x, :modes_y].unsqueeze(0) + delta

    def forward(
        self,
        x: torch.Tensor,
        condition_embed: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"2D operator expects [B, O, X, Y], got {tuple(x.shape)}.")
        x_ft = torch.fft.rfft2(x)
        if weights is None:
            weights = self.make_weights(x, condition_embed)
        modes_x = weights.shape[1]
        modes_y = weights.shape[2]
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :modes_x, :modes_y] = self._time_march(
            x_ft[:, :, :modes_x, :modes_y], weights
        )
        out_ft[:, :, -modes_x:, :modes_y] = self._time_march(
            x_ft[:, :, -modes_x:, :modes_y], weights
        )
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])
