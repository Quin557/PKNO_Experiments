"""Stable condition-parameterized Fourier Koopman operators for PKNO-U."""

from __future__ import annotations

import torch
from torch import nn

from pkno.operators.koopman_parameterized import KoopmanMatrixGenerator
from pkno_u.stability import ContractiveTransition


class StableParameterizedKoopmanOperator1D(nn.Module):
    def __init__(
        self,
        *,
        observable_dim: int,
        modes: int,
        condition_embed_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
        delta_scale: float = 0.05,
        max_operator_norm: float = 0.98,
    ) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (observable_dim * observable_dim)
        self.base_matrix = nn.Parameter(
            scale * torch.randn(modes, observable_dim, observable_dim, dtype=torch.cfloat)
        )
        self.generator = KoopmanMatrixGenerator(
            freq_dim=1,
            condition_embed_dim=condition_embed_dim,
            observable_dim=observable_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            delta_scale=delta_scale,
        )
        self.transition = ContractiveTransition(max_operator_norm)

    def make_weights(self, x: torch.Tensor, condition_embed: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x)
        modes = min(self.modes, x_ft.shape[-1])
        freq = (torch.arange(modes, device=x.device, dtype=x.real.dtype) / max(modes - 1, 1)).unsqueeze(-1)
        residual = self.base_matrix[:modes].unsqueeze(0) + self.generator(freq, condition_embed)
        return self.transition(residual)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros_like(x_ft)
        modes = weights.shape[1]
        out_ft[:, :, :modes] = torch.einsum("bim,bmio->bom", x_ft[:, :, :modes], weights)
        return torch.fft.irfft(out_ft, n=x.shape[-1])


class StableParameterizedKoopmanOperator2D(nn.Module):
    def __init__(
        self,
        *,
        observable_dim: int,
        modes_x: int,
        modes_y: int,
        condition_embed_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
        delta_scale: float = 0.05,
        max_operator_norm: float = 0.98,
    ) -> None:
        super().__init__()
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
            hidden_dim=hidden_dim,
            depth=depth,
            delta_scale=delta_scale,
        )
        self.transition = ContractiveTransition(max_operator_norm)

    def make_weights(self, x: torch.Tensor, condition_embed: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft2(x)
        modes_x = min(self.modes_x, x_ft.shape[-2])
        modes_y = min(self.modes_y, x_ft.shape[-1])
        kx = torch.arange(modes_x, device=x.device, dtype=x.real.dtype) / max(modes_x - 1, 1)
        ky = torch.arange(modes_y, device=x.device, dtype=x.real.dtype) / max(modes_y - 1, 1)
        grid_x, grid_y = torch.meshgrid(kx, ky, indexing="ij")
        freq = torch.stack([grid_x, grid_y], dim=-1)
        residual = self.base_matrix[:modes_x, :modes_y].unsqueeze(0) + self.generator(freq, condition_embed)
        return self.transition(residual)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)
        modes_x, modes_y = weights.shape[1:3]
        out_ft[:, :, :modes_x, :modes_y] = torch.einsum(
            "bixy,bxyio->boxy", x_ft[:, :, :modes_x, :modes_y], weights
        )
        out_ft[:, :, -modes_x:, :modes_y] = torch.einsum(
            "bixy,bxyio->boxy", x_ft[:, :, -modes_x:, :modes_y], weights
        )
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])
