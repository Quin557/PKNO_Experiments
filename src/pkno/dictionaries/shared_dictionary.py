"""Shared observable dictionaries for Parameterized-KNO.

The TensorFlow PKNN reference learns a parameter-independent dictionary
``Psi(x)`` and lets the Koopman matrix vary with the physical/control
condition.  This module keeps that split in PyTorch: conditions are not fed
into the dictionary, so every dataset condition is represented in the same
observable coordinates before the parameterized Koopman update is applied.
"""

from __future__ import annotations

import torch
from torch import nn


class SharedPointwiseDictionary(nn.Module):
    """Pointwise shared dictionary for channels-last PDE fields.

    Inputs use channels-last layout:

    - 1D: ``[B, X, C]``
    - 2D: ``[B, X, Y, C]``

    The output has the same spatial grid and ``observable_dim`` channels.  A
    learned physical lift is added to a trainable nonlinear observable block.
    This is the PyTorch analogue of PKNN's ``[state, NN(state)]`` dictionary,
    adapted to KNO fields instead of flat ODE states.
    """

    def __init__(
        self,
        input_dim: int,
        observable_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or observable_dim <= 0:
            raise ValueError("input_dim and observable_dim must be positive.")
        if depth < 1:
            raise ValueError("depth must be at least 1.")

        self.input_dim = input_dim
        self.observable_dim = observable_dim
        self.physical_lift = nn.Linear(input_dim, observable_dim)

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, observable_dim))
        self.learned_observables = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {x.shape[-1]} for shape {tuple(x.shape)}."
            )
        return torch.tanh(self.physical_lift(x) + self.learned_observables(x))


class PointwiseDecoder(nn.Module):
    """Decode shared observable fields back to channels-last physical fields."""

    def __init__(
        self,
        observable_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        depth: int = 1,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if observable_dim <= 0 or output_dim <= 0:
            raise ValueError("observable_dim and output_dim must be positive.")

        layers: list[nn.Module] = []
        in_dim = observable_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class StateSummaryEncoder(nn.Module):
    """Build a compact dynamic-condition embedding from the current history.

    PKNN's parameter ``u`` may be static or time-varying.  For PDE datasets, the
    explicit condition vector can be incomplete, so Stage3_0 also constructs a
    sample- and step-dependent condition summary from the current rollout
    history.  This is not part of the shared dictionary itself; it only
    conditions the generated Koopman matrix ``K_k``.

    Features per input channel:

    - mean, standard deviation, RMS energy;
    - finite-difference gradient RMS;
    - boundary mean and standard deviation;
    - low/mid/high spectral energy ratios.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.features_per_channel = 9
        self.net = nn.Sequential(
            nn.Linear(self.features_per_channel * input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, embed_dim),
            activation(),
        )

    @staticmethod
    def _gradient_rms(x: torch.Tensor, spatial_dims: tuple[int, ...]) -> torch.Tensor:
        terms = []
        for dim in spatial_dims:
            if x.shape[dim] > 1:
                terms.append(torch.diff(x, dim=dim).square().mean(dim=spatial_dims).sqrt())
        if not terms:
            return torch.zeros_like(x.mean(dim=spatial_dims))
        return torch.stack(terms, dim=0).mean(dim=0)

    @staticmethod
    def _boundary_stats(x: torch.Tensor, spatial_dims: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        slices = []
        for dim in spatial_dims:
            first = x.select(dim, 0)
            last = x.select(dim, x.shape[dim] - 1)
            slices.extend([first, last])
        if not slices:
            base = x.mean(dim=spatial_dims)
            return base, torch.zeros_like(base)
        boundary = torch.cat([item.reshape(item.shape[0], -1, item.shape[-1]) for item in slices], dim=1)
        return boundary.mean(dim=1), boundary.std(dim=1, unbiased=False)

    @staticmethod
    def _spectral_ratios(x: torch.Tensor, spatial_dims: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_ft = torch.fft.fftn(x, dim=spatial_dims)
        energy = x_ft.abs().square()
        spatial_shape = tuple(x.shape[dim] for dim in spatial_dims)
        grids = torch.meshgrid(
            *[torch.fft.fftfreq(size, device=x.device, dtype=x.dtype) for size in spatial_shape],
            indexing="ij",
        )
        radius = torch.zeros(spatial_shape, device=x.device, dtype=x.dtype)
        for grid in grids:
            radius = radius + grid.square()
        radius = radius.sqrt()
        radius = radius / radius.max().clamp_min(1e-12)

        view_shape = [1] * x.ndim
        for axis, size in zip(spatial_dims, spatial_shape):
            view_shape[axis] = size
        radius = radius.reshape(view_shape)
        total = energy.sum(dim=spatial_dims).clamp_min(1e-12)
        low = energy.masked_fill(~((radius >= 0.0) & (radius < 1.0 / 3.0)), 0).sum(dim=spatial_dims) / total
        mid = energy.masked_fill(~((radius >= 1.0 / 3.0) & (radius < 2.0 / 3.0)), 0).sum(dim=spatial_dims) / total
        high = energy.masked_fill(~((radius >= 2.0 / 3.0) & (radius <= 1.000001)), 0).sum(dim=spatial_dims) / total
        return low, mid, high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {x.shape[-1]} for shape {tuple(x.shape)}."
            )
        spatial_dims = tuple(range(1, x.ndim - 1))
        mean = x.mean(dim=spatial_dims)
        std = x.std(dim=spatial_dims, unbiased=False)
        rms = x.square().mean(dim=spatial_dims).sqrt()
        grad_rms = self._gradient_rms(x, spatial_dims)
        boundary_mean, boundary_std = self._boundary_stats(x, spatial_dims)
        low_ratio, mid_ratio, high_ratio = self._spectral_ratios(x, spatial_dims)
        features = torch.cat(
            [
                mean,
                std,
                rms,
                grad_rms,
                boundary_mean,
                boundary_std,
                low_ratio,
                mid_ratio,
                high_ratio,
            ],
            dim=-1,
        )
        return self.net(features)
