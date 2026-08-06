"""Physics-first, normalized condition path for PKNO_v2."""

from __future__ import annotations

import torch
from torch import nn


class CompactStateEncoder(nn.Module):
    """A small, bounded state summary that cannot dominate physical metadata."""

    def __init__(self, input_dim: int, embed_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(nn.Linear(4 * input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = tuple(range(1, x.ndim - 1))
        mean = x.mean(dim=spatial)
        std = x.std(dim=spatial, unbiased=False)
        rms = x.square().mean(dim=spatial).sqrt()
        temporal = (x - x.mean(dim=-1, keepdim=True)).mean(dim=spatial)
        return self.net(torch.cat([mean, std, rms, temporal], dim=-1))


class PhysicsFirstConditioner(nn.Module):
    """Normalize c_static and add a bounded low-dimensional state correction."""

    def __init__(self, condition_dim: int, input_dim: int, embed_dim: int = 64, state_dim: int = 16,
                 hidden_dim: int = 128, state_gate_max: float = 0.15) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.state_gate_max = state_gate_max
        self.register_buffer("condition_scale", torch.ones(condition_dim))
        self.physics = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim), nn.GELU())
        self.state_encoder = CompactStateEncoder(input_dim, state_dim, min(hidden_dim, 64))
        self.state = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.Sigmoid())

    def set_condition_scale(self, scale: torch.Tensor) -> None:
        if tuple(scale.shape) != tuple(self.condition_scale.shape):
            raise ValueError("condition scale shape mismatch")
        self.condition_scale.copy_(scale.detach().float().clamp_min(1e-3))

    def forward(self, condition: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if condition.ndim != 2 or condition.shape[-1] != self.condition_dim:
            raise ValueError(f"condition must be [B,{self.condition_dim}], got {tuple(condition.shape)}")
        physics = self.physics(condition / self.condition_scale)
        state = self.state(self.state_encoder(history))
        gate = self.state_gate_max * self.gate(torch.cat([physics, state], dim=-1))
        return physics + gate * state, gate
