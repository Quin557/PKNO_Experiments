"""Memory-bounded soft-stability terms for PKNO_v1."""

from __future__ import annotations

import torch


def rms_ratio(next_field: torch.Tensor, current_field: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    dims = tuple(range(1, next_field.ndim))
    return next_field.square().mean(dim=dims).sqrt() / current_field.square().mean(dim=dims).sqrt().clamp_min(eps)


def state_correction_penalty(correction: torch.Tensor) -> torch.Tensor:
    return correction.square().mean()


def temporal_state_penalty(current: torch.Tensor, previous_detached: torch.Tensor | None) -> torch.Tensor:
    if previous_detached is None:
        return current.new_zeros(())
    return (current - previous_detached).square().mean()


def growth_envelope_penalty(prediction: torch.Tensor, current: torch.Tensor, ceiling: float) -> torch.Tensor:
    return torch.relu(rms_ratio(prediction, current) - ceiling).square().mean()
