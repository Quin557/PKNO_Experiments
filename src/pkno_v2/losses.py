"""PKNO_v2 auxiliary losses kept deliberately small and dataset agnostic."""

from __future__ import annotations

import torch


def gradient_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, pred.ndim - 1))
    if not dims:
        return pred.new_zeros(())
    terms = []
    for dim in dims:
        if pred.shape[dim] > 1:
            terms.append((torch.diff(pred, dim=dim) - torch.diff(target, dim=dim)).square().mean())
    return torch.stack(terms).mean() if terms else pred.new_zeros(())


def high_frequency_mse(pred: torch.Tensor, target: torch.Tensor, cutoff: float = 0.5) -> torch.Tensor:
    dims = tuple(range(1, pred.ndim - 1))
    if not dims:
        return pred.new_zeros(())
    pred_ft = torch.fft.fftn(pred.float(), dim=dims, norm="ortho")
    target_ft = torch.fft.fftn(target.float(), dim=dims, norm="ortho")
    shape = tuple(pred.shape[d] for d in dims)
    grids = torch.meshgrid(*[torch.fft.fftfreq(n, device=pred.device) for n in shape], indexing="ij")
    radius = torch.zeros(shape, device=pred.device)
    for grid in grids:
        radius = radius + grid.square()
    mask = radius.sqrt() >= cutoff * radius.sqrt().max().clamp_min(1e-12)
    view = [1] * pred.ndim
    for axis, size in zip(dims, shape):
        view[axis] = size
    mask = mask.reshape(view)
    return ((pred_ft - target_ft).abs().square().masked_select(mask.expand_as(pred_ft))).mean()


def late_step_weights(horizon: int, final_weight: float = 2.0, device: torch.device | None = None) -> torch.Tensor:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return torch.linspace(1.0, final_weight, horizon, device=device)
