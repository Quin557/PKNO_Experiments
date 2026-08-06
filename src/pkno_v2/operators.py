"""Low-rank, softly bounded parameterized Fourier Koopman operators."""

from __future__ import annotations

import torch
from torch import nn


class LowRankKoopman(nn.Module):
    """K_k(c)=K0_k+U_k diag(alpha(c)) V_k^H with a bounded Euler step."""

    def __init__(self, spatial_dim: int, observable_dim: int, modes: int = 16, condition_dim: int = 64,
                 rank: int = 4, hidden_dim: int = 128, depth: int = 2, delta_scale: float = 0.02,
                 eta_max: float = 0.5) -> None:
        super().__init__()
        self.spatial_dim, self.observable_dim, self.modes, self.rank = spatial_dim, observable_dim, modes, rank
        self.delta_scale, self.eta_max = delta_scale, eta_max
        shape = (modes, observable_dim, observable_dim) if spatial_dim == 1 else (modes, modes, observable_dim, observable_dim)
        scale = 0.02 / max(observable_dim, 1)
        self.base_real = nn.Parameter(scale * torch.randn(*shape))
        self.base_imag = nn.Parameter(scale * torch.randn(*shape))
        vec_shape = (modes, observable_dim, rank) if spatial_dim == 1 else (modes, modes, observable_dim, rank)
        self.u_real = nn.Parameter(scale * torch.randn(*vec_shape)); self.u_imag = nn.Parameter(scale * torch.randn(*vec_shape))
        self.v_real = nn.Parameter(scale * torch.randn(*vec_shape)); self.v_imag = nn.Parameter(scale * torch.randn(*vec_shape))
        in_dim = condition_dim + (1 if spatial_dim == 1 else 2)
        layers: list[nn.Module] = []
        current = in_dim
        for _ in range(max(depth, 1)):
            layers += [nn.Linear(current, hidden_dim), nn.GELU()]; current = hidden_dim
        layers.append(nn.Linear(current, 2 * rank))
        self.alpha = nn.Sequential(*layers)
        self.raw_eta = nn.Parameter(torch.tensor(-2.0))

    def _freq(self, mx: int, my: int | None, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.spatial_dim == 1:
            return (torch.arange(mx, device=device, dtype=dtype) / max(mx - 1, 1)).unsqueeze(-1)
        kx = torch.arange(mx, device=device, dtype=dtype) / max(mx - 1, 1)
        ky = torch.arange(my or mx, device=device, dtype=dtype) / max((my or mx) - 1, 1)
        gx, gy = torch.meshgrid(kx, ky, indexing="ij")
        return torch.stack([gx, gy], dim=-1)

    def make_weights(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        ft = torch.fft.rfft(x) if self.spatial_dim == 1 else torch.fft.rfft2(x)
        mx = min(self.modes, ft.shape[-1] if self.spatial_dim == 1 else ft.shape[-2])
        my = None if self.spatial_dim == 1 else min(self.modes, ft.shape[-1])
        freq = self._freq(mx, my, x.device, x.real.dtype)
        raw = self.alpha(torch.cat([freq.reshape(-1, freq.shape[-1]).unsqueeze(0).expand(cond.shape[0], -1, -1), cond.unsqueeze(1).expand(-1, freq.reshape(-1, freq.shape[-1]).shape[0], -1)], -1))
        raw = raw.reshape(cond.shape[0], *freq.shape[:-1], self.rank, 2)
        alpha = self.delta_scale * torch.complex(raw[..., 0], raw[..., 1])
        base = torch.complex(self.base_real, self.base_imag)
        u = torch.complex(self.u_real, self.u_imag); v = torch.complex(self.v_real, self.v_imag)
        if self.spatial_dim == 1:
            return base[:mx].unsqueeze(0) + torch.einsum("bmr,mor,mqr->bmoq", alpha, u[:mx], torch.conj(v[:mx]))
        return base[:mx, :my].unsqueeze(0) + torch.einsum("bxyr,xyor,xyqr->bxyoq", alpha, u[:mx, :my], torch.conj(v[:mx, :my]))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        if weights is None:
            weights = self.make_weights(x, cond)
        ft = torch.fft.rfft(x) if self.spatial_dim == 1 else torch.fft.rfft2(x)
        if self.spatial_dim == 1:
            m = weights.shape[1]; out = torch.zeros_like(ft)
            out[:, :, :m] = torch.einsum("bim,bmio->bom", ft[:, :, :m], weights)
            return torch.fft.irfft(out, n=x.shape[-1])
        mx, my = weights.shape[1], weights.shape[2]; out = torch.zeros_like(ft)
        out[:, :, :mx, :my] = torch.einsum("bixy,bxyio->boxy", ft[:, :, :mx, :my], weights)
        out[:, :, -mx:, :my] = torch.einsum("bixy,bxyio->boxy", ft[:, :, -mx:, :my], weights)
        return torch.fft.irfft2(out, s=x.shape[-2:])

    def step(self, x: torch.Tensor, cond: torch.Tensor, weights: torch.Tensor, step_scale: float = 1.0) -> torch.Tensor:
        eta = self.eta_max * torch.sigmoid(self.raw_eta)
        return x + (eta * step_scale) * self.forward(x, cond, weights)
