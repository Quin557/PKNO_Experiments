"""Guaranteed norm bounds for generated complex Koopman matrices."""

from __future__ import annotations

import torch
from torch import nn


class ContractiveTransition(nn.Module):
    """Turn an unconstrained residual matrix into a contractive transition.

    ``rho * (I + D) / (1 + ||D||_F)`` has spectral norm at most ``rho`` while
    retaining an identity-centered parameterization.  This avoids an expensive
    SVD for every batch-specific Fourier mode.
    """

    def __init__(self, max_norm: float = 0.98, eps: float = 1e-12) -> None:
        super().__init__()
        if max_norm <= 0:
            raise ValueError("max_norm must be positive.")
        self.max_norm = float(max_norm)
        self.eps = float(eps)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape[-1] != residual.shape[-2]:
            raise ValueError("Koopman residual matrices must be square.")
        dim = residual.shape[-1]
        identity = torch.eye(dim, device=residual.device, dtype=residual.dtype)
        norm = torch.linalg.vector_norm(residual, dim=(-2, -1), keepdim=True)
        return self.max_norm * (identity + residual) / (1.0 + norm.clamp_min(self.eps))

    @staticmethod
    def residual_norms(residual: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(residual, dim=(-2, -1))
