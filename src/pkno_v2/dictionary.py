"""Condition-independent observable dictionary for PKNO_v2.

The implementation intentionally stays small: fixed local observables preserve
the PKNN-style [1, state, nonlinear] split while the learned part is a
pointwise MLP.  Inputs and outputs are channels-last, matching the Stage 3
data protocol.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[dim] < 2:
        return torch.zeros_like(x)
    out = torch.zeros_like(x)
    a = [slice(None)] * x.ndim
    b = [slice(None)] * x.ndim
    a[dim], b[dim] = 0, 1
    out[tuple(a)] = x[tuple(b)] - x[tuple(a)]
    a[dim], b[dim] = -1, -2
    out[tuple(a)] = x[tuple(a)] - x[tuple(b)]
    if x.shape[dim] > 2:
        m, p, n = [slice(None)] * x.ndim, [slice(None)] * x.ndim, [slice(None)] * x.ndim
        m[dim], p[dim], n[dim] = slice(1, -1), slice(2, None), slice(None, -2)
        out[tuple(m)] = 0.5 * (x[tuple(p)] - x[tuple(n)])
    return out


class ObservableDictionary(nn.Module):
    """Shared local dictionary with deterministic physics-inspired features."""

    def __init__(self, input_dim: int, observable_dim: int = 32, hidden_dim: int = 128,
                 depth: int = 2, basis_kind: str = "generic") -> None:
        super().__init__()
        if observable_dim < 8:
            raise ValueError("observable_dim must be at least 8")
        self.input_dim = input_dim
        self.observable_dim = observable_dim
        self.basis_kind = basis_kind
        # Keep a fixed eight-channel basis for every dataset so O=32 is exact.
        fixed_dim = 8
        self.fixed_dim = min(fixed_dim, observable_dim)
        learned_dim = observable_dim - self.fixed_dim
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(max(depth, 1)):
            layers += [nn.Linear(current, hidden_dim), nn.GELU()]
            current = hidden_dim
        if learned_dim:
            layers.append(nn.Linear(current, learned_dim))
        self.learned = nn.Sequential(*layers) if layers else nn.Identity()

    def _fixed(self, x: torch.Tensor) -> torch.Tensor:
        latest = x[..., -1:]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, unbiased=False, keepdim=True)
        parts = [torch.ones_like(latest), latest, latest.square(), mean, std]
        if x.ndim == 3:
            ux = _diff(latest, 1)
            parts += [ux, ux.square(), torch.sin(math.pi * latest)]
        else:
            ux, uy = _diff(latest, 1), _diff(latest, 2)
            grad = torch.sqrt(ux.square() + uy.square() + 1e-12)
            lap = _diff(_diff(latest, 1), 1) + _diff(_diff(latest, 2), 2)
            parts += [grad, lap, (latest - latest.mean(dim=(1, 2), keepdim=True))]
        return torch.cat(parts, dim=-1)[..., : self.fixed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected channels-last input with {self.input_dim} channels, got {tuple(x.shape)}")
        fixed = self._fixed(x)
        learned = torch.tanh(self.learned(x))
        return torch.cat([fixed, learned], dim=-1)


class PointwiseDecoder(nn.Module):
    def __init__(self, observable_dim: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observable_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
