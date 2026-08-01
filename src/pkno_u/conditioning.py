"""Condition encoders for the Stage3_1 PKNO-U ablations.

The shared dictionary deliberately remains condition independent.  This module
only produces the vector used by the Koopman-matrix generator.
"""

from __future__ import annotations

import torch
from torch import nn


CONDITION_MODES = ("physical_only", "physical_compact_state", "physical_gated_state")


def _spatial_features(history: torch.Tensor) -> torch.Tensor:
    """Return low-dimensional, normalized slow statistics for each history channel."""

    spatial_dims = tuple(range(1, history.ndim - 1))
    mean = history.mean(dim=spatial_dims)
    std = history.std(dim=spatial_dims, unbiased=False)
    rms = history.square().mean(dim=spatial_dims).sqrt()

    gradients: list[torch.Tensor] = []
    for dim in spatial_dims:
        if history.shape[dim] > 1:
            gradients.append(torch.diff(history, dim=dim).square().mean(dim=spatial_dims).sqrt())
    gradient_rms = torch.stack(gradients, dim=0).mean(dim=0) if gradients else torch.zeros_like(mean)

    x_ft = torch.fft.fftn(history, dim=spatial_dims)
    energy = x_ft.abs().square()
    spatial_shape = tuple(history.shape[dim] for dim in spatial_dims)
    grids = torch.meshgrid(
        *[torch.fft.fftfreq(size, device=history.device, dtype=history.dtype) for size in spatial_shape],
        indexing="ij",
    )
    radius = torch.zeros(spatial_shape, device=history.device, dtype=history.dtype)
    for grid in grids:
        radius = radius + grid.square()
    radius = radius.sqrt()
    radius = radius / radius.max().clamp_min(1e-12)
    view_shape = [1] * history.ndim
    for dim, size in zip(spatial_dims, spatial_shape):
        view_shape[dim] = size
    radius = radius.reshape(view_shape)
    total = energy.sum(dim=spatial_dims).clamp_min(1e-12)
    low_ratio = energy.masked_fill(radius >= 1.0 / 3.0, 0.0).sum(dim=spatial_dims) / total
    high_ratio = energy.masked_fill(radius < 2.0 / 3.0, 0.0).sum(dim=spatial_dims) / total
    return torch.cat([mean, std, rms, gradient_rms, low_ratio, high_ratio], dim=-1)


class ConditioningModule(nn.Module):
    """Encode physical conditions with optional constrained state adaptation."""

    def __init__(
        self,
        *,
        condition_dim: int,
        input_channels: int,
        output_dim: int,
        mode: str = "physical_only",
        hidden_dim: int = 128,
        state_dim: int = 16,
    ) -> None:
        super().__init__()
        if mode not in CONDITION_MODES:
            raise ValueError(f"Unknown condition mode {mode!r}; expected one of {CONDITION_MODES}.")
        if condition_dim <= 0 or input_channels <= 0 or output_dim <= 0:
            raise ValueError("condition_dim, input_channels, and output_dim must be positive.")
        self.mode = mode
        self.condition_dim = condition_dim
        self.input_channels = input_channels
        self.output_dim = output_dim

        self.physical_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )
        self.state_encoder: nn.Module | None = None
        self.state_to_output: nn.Module | None = None
        self.gate: nn.Module | None = None
        if mode != "physical_only":
            self.state_encoder = nn.Sequential(
                nn.LayerNorm(6 * input_channels),
                nn.Linear(6 * input_channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, state_dim),
                nn.Tanh(),
            )
            self.state_to_output = nn.Sequential(nn.Linear(state_dim, output_dim), nn.Tanh())
            if mode == "physical_gated_state":
                self.gate = nn.Sequential(nn.Linear(state_dim, 1), nn.Sigmoid())

    def forward(self, condition: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if condition.ndim != 2 or condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"Expected physical condition [B, {self.condition_dim}], got {tuple(condition.shape)}."
            )
        if history.shape[0] != condition.shape[0] or history.shape[-1] != self.input_channels:
            raise ValueError("Condition and history must have matching batch size and configured history channels.")
        physical = self.physical_encoder(condition)
        if self.mode == "physical_only":
            return physical, physical.new_zeros((physical.shape[0], 1))

        if self.state_encoder is None or self.state_to_output is None:
            raise RuntimeError("State condition modules were not initialized.")
        state = self.state_encoder(_spatial_features(history))
        update = self.state_to_output(state)
        if self.mode == "physical_compact_state":
            return physical + update, physical.new_ones((physical.shape[0], 1))
        if self.gate is None:
            raise RuntimeError("Gate module was not initialized.")
        gate = self.gate(state)
        return physical + gate * update, gate
