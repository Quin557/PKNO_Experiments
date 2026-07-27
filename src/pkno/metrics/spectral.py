from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpectralBands:
    low: tuple[float, float] = (0.0, 1.0 / 3.0)
    mid: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    high: tuple[float, float] = (2.0 / 3.0, 1.000001)


def _spatial_dims(x: torch.Tensor, dims: tuple[int, ...] | None) -> tuple[int, ...]:
    if dims is not None:
        return dims
    if x.ndim < 3:
        raise ValueError("Expected at least batch, space, channel dimensions.")
    return tuple(range(1, x.ndim - 1))


def _frequency_radius(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    grids = []
    for n in shape:
        freq = torch.fft.fftfreq(n, device=device)
        grids.append(freq)
    mesh = torch.meshgrid(*grids, indexing="ij")
    radius = torch.zeros(shape, device=device)
    for grid in mesh:
        radius = radius + grid.square()
    radius = radius.sqrt()
    max_radius = radius.max().clamp_min(1e-12)
    return radius / max_radius


def spectral_band_relative_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    dims: tuple[int, ...] | None = None,
    bands: SpectralBands = SpectralBands(),
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Compute relative L2 error in low, mid, and high FFT bands.

    Tensors are expected to use channels-last layout, for example `[B, X, C]`
    or `[B, X, Y, C]`. Pass `dims` explicitly for a different layout.
    """

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    spatial_dims = _spatial_dims(pred, dims)
    pred_fft = torch.fft.fftn(pred, dim=spatial_dims)
    target_fft = torch.fft.fftn(target, dim=spatial_dims)
    err_fft = pred_fft - target_fft

    spatial_shape = tuple(pred.shape[d] for d in spatial_dims)
    radius = _frequency_radius(spatial_shape, pred.device)

    view_shape = [1] * pred.ndim
    for axis, size in zip(spatial_dims, spatial_shape):
        view_shape[axis] = size
    radius = radius.reshape(view_shape)

    out: dict[str, torch.Tensor] = {}
    for name, (lo, hi) in bands.__dict__.items():
        mask = (radius >= lo) & (radius < hi)
        err_norm = torch.linalg.vector_norm(err_fft.masked_select(mask))
        target_norm = torch.linalg.vector_norm(target_fft.masked_select(mask))
        out[f"{name}_band_spectral_rel_l2"] = err_norm / target_norm.clamp_min(eps)

    pred_energy = pred_fft.abs().square()
    target_energy = target_fft.abs().square()
    high_mask = (radius >= bands.high[0]) & (radius < bands.high[1])
    pred_high_ratio = pred_energy.masked_select(high_mask).sum() / pred_energy.sum().clamp_min(eps)
    target_high_ratio = target_energy.masked_select(high_mask).sum() / target_energy.sum().clamp_min(eps)
    out["high_frequency_energy_ratio_error"] = (pred_high_ratio - target_high_ratio).abs()

    return out


def gradient_relative_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    dims: tuple[int, ...] | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compare finite-difference gradients over spatial dimensions."""

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    spatial_dims = _spatial_dims(pred, dims)
    err_terms = []
    target_terms = []
    for dim in spatial_dims:
        pred_grad = torch.diff(pred, dim=dim)
        target_grad = torch.diff(target, dim=dim)
        err_terms.append(torch.linalg.vector_norm(pred_grad - target_grad).square())
        target_terms.append(torch.linalg.vector_norm(target_grad).square())

    err = torch.stack(err_terms).sum().sqrt()
    denom = torch.stack(target_terms).sum().sqrt().clamp_min(eps)
    return err / denom
