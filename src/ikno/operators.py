"""Fixed low-frequency Fourier Koopman operators for IKNO."""

from __future__ import annotations

import torch
from torch import nn


class FixedKoopmanOperator1D(nn.Module):
    """One shared complex Koopman matrix for each retained 1D Fourier mode."""

    def __init__(self, observable_dim: int, modes: int) -> None:
        super().__init__()
        if observable_dim <= 0 or modes <= 0:
            raise ValueError("observable_dim and modes must be positive.")
        scale = 1.0 / (observable_dim * observable_dim)
        self.modes = modes
        self.matrix = nn.Parameter(
            scale * torch.randn(modes, observable_dim, observable_dim, dtype=torch.cfloat)
        )

    @staticmethod
    def _apply_matrix(x_ft: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bim,mio->bom", x_ft, matrix)

    def forward(self, x: torch.Tensor, *, power: int = 1) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, O, X], got {tuple(x.shape)}.")
        if power <= 0:
            raise ValueError("power must be positive.")
        x_ft = torch.fft.rfft(x)
        modes = min(self.modes, x_ft.shape[-1])
        evolved = x_ft[:, :, :modes]
        for _ in range(power):
            evolved = self._apply_matrix(evolved, self.matrix[:modes])
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :modes] = evolved
        return torch.fft.irfft(out_ft, n=x.shape[-1])


class FixedKoopmanOperator2D(nn.Module):
    """One shared complex Koopman matrix for each retained 2D Fourier mode."""

    def __init__(self, observable_dim: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        if observable_dim <= 0 or modes_x <= 0 or modes_y <= 0:
            raise ValueError("observable_dim and mode counts must be positive.")
        scale = 1.0 / (observable_dim * observable_dim)
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.matrix = nn.Parameter(
            scale
            * torch.randn(
                modes_x,
                modes_y,
                observable_dim,
                observable_dim,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def _apply_matrix(x_ft: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,xyio->boxy", x_ft, matrix)

    def _evolve(self, x_ft: torch.Tensor, matrix: torch.Tensor, power: int) -> torch.Tensor:
        evolved = x_ft
        for _ in range(power):
            evolved = self._apply_matrix(evolved, matrix)
        return evolved

    def forward(self, x: torch.Tensor, *, power: int = 1) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, O, X, Y], got {tuple(x.shape)}.")
        if power <= 0:
            raise ValueError("power must be positive.")
        x_ft = torch.fft.rfft2(x)
        modes_x = min(self.modes_x, x_ft.shape[-2])
        modes_y = min(self.modes_y, x_ft.shape[-1])
        matrix = self.matrix[:modes_x, :modes_y]
        out_ft = torch.zeros_like(x_ft)
        positive = self._evolve(x_ft[:, :, :modes_x, :modes_y], matrix, power)
        negative = self._evolve(x_ft[:, :, -modes_x:, :modes_y], matrix, power)
        out_ft[:, :, :modes_x, :modes_y] = positive
        out_ft[:, :, -modes_x:, :modes_y] = negative
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])
