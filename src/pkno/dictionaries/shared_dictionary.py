"""Shared observable dictionaries for Parameterized-KNO.

The TensorFlow PKNN reference learns a parameter-independent dictionary
``Psi(x)`` and lets the Koopman matrix vary with the physical/control
condition.  This module keeps that split in PyTorch: conditions are not fed
into the dictionary, so every dataset condition is represented in the same
observable coordinates before the parameterized Koopman update is applied.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class HandcraftedObservableBasis(nn.Module):
    """Dataset-aware fixed basis functions for channels-last PDE fields.

    PKNN's default dictionary is explicitly ``[1, x, DicNN(x)]``.  For PDE
    fields, the same idea is more useful when ``x`` is augmented with a few
    local physics observables before the learned block.  These features are
    still condition-independent, so they preserve the shared-dictionary
    assumption.
    """

    _OUTPUT_DIMS = {
        "generic": 4,
        "burgers": 9,
        "navier_stokes": 10,
        "shallow_water": 10,
    }

    def __init__(self, input_dim: int, basis_kind: str = "generic") -> None:
        super().__init__()
        if basis_kind not in self._OUTPUT_DIMS:
            allowed = ", ".join(sorted(self._OUTPUT_DIMS))
            raise ValueError(f"Unknown basis_kind {basis_kind!r}; expected one of: {allowed}.")
        self.input_dim = input_dim
        self.basis_kind = basis_kind
        self.output_dim = self._OUTPUT_DIMS[basis_kind]

    @staticmethod
    def _latest(x: torch.Tensor) -> torch.Tensor:
        return x[..., -1:]

    @staticmethod
    def _history_std(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 1:
            return torch.zeros_like(x[..., :1])
        return x.std(dim=-1, unbiased=False, keepdim=True)

    @staticmethod
    def _finite_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[dim] <= 1:
            return torch.zeros_like(x)

        out = torch.zeros_like(x)

        first_out = [slice(None)] * x.ndim
        first_out[dim] = 0
        first_next = [slice(None)] * x.ndim
        first_next[dim] = 1
        out[tuple(first_out)] = x[tuple(first_next)] - x[tuple(first_out)]

        last_out = [slice(None)] * x.ndim
        last_out[dim] = x.shape[dim] - 1
        last_prev = [slice(None)] * x.ndim
        last_prev[dim] = x.shape[dim] - 2
        out[tuple(last_out)] = x[tuple(last_out)] - x[tuple(last_prev)]

        if x.shape[dim] > 2:
            mid = [slice(None)] * x.ndim
            prev = [slice(None)] * x.ndim
            nxt = [slice(None)] * x.ndim
            mid[dim] = slice(1, -1)
            prev[dim] = slice(0, -2)
            nxt[dim] = slice(2, None)
            out[tuple(mid)] = 0.5 * (x[tuple(nxt)] - x[tuple(prev)])

        return out

    @staticmethod
    def _second_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[dim] <= 2:
            return torch.zeros_like(x)

        out = torch.zeros_like(x)
        mid = [slice(None)] * x.ndim
        prev = [slice(None)] * x.ndim
        nxt = [slice(None)] * x.ndim
        mid[dim] = slice(1, -1)
        prev[dim] = slice(0, -2)
        nxt[dim] = slice(2, None)
        out[tuple(mid)] = x[tuple(nxt)] - 2.0 * x[tuple(mid)] + x[tuple(prev)]
        return out

    @staticmethod
    def _boundary_mask_like(x: torch.Tensor, spatial_dims: tuple[int, ...]) -> torch.Tensor:
        mask = torch.zeros_like(x)
        for dim in spatial_dims:
            first = [slice(None)] * x.ndim
            last = [slice(None)] * x.ndim
            first[dim] = 0
            last[dim] = x.shape[dim] - 1
            mask[tuple(first)] = 1.0
            mask[tuple(last)] = 1.0
        return mask

    def _generic(self, x: torch.Tensor) -> torch.Tensor:
        latest = self._latest(x)
        return torch.cat(
            [
                torch.ones_like(latest),
                latest,
                x.mean(dim=-1, keepdim=True),
                self._history_std(x),
            ],
            dim=-1,
        )

    def _burgers(self, x: torch.Tensor) -> torch.Tensor:
        u = self._latest(x)
        ux = self._finite_diff(u, dim=1)
        uxx = self._second_diff(u, dim=1)
        return torch.cat(
            [
                torch.ones_like(u),
                u,
                u.square(),
                u.square() * u,
                torch.sin(math.pi * u),
                torch.cos(math.pi * u),
                ux,
                uxx,
                u * ux,
            ],
            dim=-1,
        )

    def _navier_stokes(self, x: torch.Tensor) -> torch.Tensor:
        w = self._latest(x)
        history_mean = x.mean(dim=-1, keepdim=True)
        history_std = self._history_std(x)
        dt_field = w - x[..., :1]
        wx = self._finite_diff(w, dim=1)
        wy = self._finite_diff(w, dim=2)
        grad_mag = torch.sqrt(wx.square() + wy.square() + 1e-12)
        lap = self._second_diff(w, dim=1) + self._second_diff(w, dim=2)
        return torch.cat(
            [
                torch.ones_like(w),
                w,
                history_mean,
                history_std,
                w.square(),
                dt_field,
                wx,
                wy,
                grad_mag,
                lap,
            ],
            dim=-1,
        )

    def _shallow_water(self, x: torch.Tensor) -> torch.Tensor:
        h = self._latest(x)
        history_mean = x.mean(dim=-1, keepdim=True)
        history_std = self._history_std(x)
        hx = self._finite_diff(h, dim=1)
        hy = self._finite_diff(h, dim=2)
        grad_mag = torch.sqrt(hx.square() + hy.square() + 1e-12)
        lap = self._second_diff(h, dim=1) + self._second_diff(h, dim=2)
        boundary = self._boundary_mask_like(h, spatial_dims=(1, 2))
        spatial_mean = h.mean(dim=(1, 2), keepdim=True)
        return torch.cat(
            [
                torch.ones_like(h),
                h,
                history_mean,
                history_std,
                h.square(),
                grad_mag,
                lap,
                boundary,
                boundary * h,
                h - spatial_mean,
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.basis_kind == "burgers":
            if x.ndim != 3:
                raise ValueError(f"Burgers basis expects [B, X, C], got {tuple(x.shape)}.")
            return self._burgers(x)
        if self.basis_kind == "navier_stokes":
            if x.ndim != 4:
                raise ValueError(f"Navier-Stokes basis expects [B, X, Y, C], got {tuple(x.shape)}.")
            return self._navier_stokes(x)
        if self.basis_kind == "shallow_water":
            if x.ndim != 4:
                raise ValueError(f"Shallow-water basis expects [B, X, Y, C], got {tuple(x.shape)}.")
            return self._shallow_water(x)
        return self._generic(x)


class SharedPointwiseDictionary(nn.Module):
    """Pointwise shared dictionary for channels-last PDE fields.

    Inputs use channels-last layout:

    - 1D: ``[B, X, C]``
    - 2D: ``[B, X, Y, C]``

    The output has the same spatial grid and ``observable_dim`` channels.  A
    The first channels are explicit handcrafted observables such as ``1``,
    ``u``, gradients, and low-order nonlinear terms.  The remaining channels
    are trainable nonlinear observables.  This is the PyTorch analogue of
    PKNN's ``[1, state, NN(state)]`` dictionary, adapted to KNO fields instead
    of flat ODE states.
    """

    def __init__(
        self,
        input_dim: int,
        observable_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
        basis_kind: str = "generic",
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or observable_dim <= 0:
            raise ValueError("input_dim and observable_dim must be positive.")
        if depth < 1:
            raise ValueError("depth must be at least 1.")

        self.input_dim = input_dim
        self.observable_dim = observable_dim
        self.basis_kind = basis_kind
        self.fixed_basis = HandcraftedObservableBasis(input_dim=input_dim, basis_kind=basis_kind)
        self.fixed_dim = self.fixed_basis.output_dim
        self.learned_dim = observable_dim - self.fixed_dim
        if self.learned_dim < 0:
            raise ValueError(
                f"observable_dim={observable_dim} is too small for {basis_kind!r} "
                f"fixed basis with {self.fixed_dim} channels."
            )

        layers: list[nn.Module] = []
        in_dim = input_dim
        if self.learned_dim > 0:
            for _ in range(depth):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(activation())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, self.learned_dim))
        self.learned_observables = nn.Sequential(*layers) if layers else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {x.shape[-1]} for shape {tuple(x.shape)}."
            )
        fixed = self.fixed_basis(x)
        if self.learned_observables is None:
            return fixed
        learned = torch.tanh(self.learned_observables(x))
        return torch.cat([fixed, learned], dim=-1)


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
