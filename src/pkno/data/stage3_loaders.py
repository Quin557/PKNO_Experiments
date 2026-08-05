"""Stage3_0 data loaders with explicit condition vectors.

The four current datasets were first used as KNO baselines.  Stage3_0 reuses
their tensors but augments every sample with a finite-dimensional condition
vector ``c`` so that ``K_k = G(k, u_embed, c)`` can be trained in PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, TensorDataset

from pkno.data.split_protocol import SampleSplit, max_index


@dataclass(frozen=True)
class LoaderBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    input_channels: int
    output_channels: int
    condition_dim: int
    t_out: int
    spatial_dim: int
    metadata: dict[str, Any]


def _constant_conditions(num: int, values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, -1).repeat(num, 1)


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return

    bad_count = int((~finite).sum().item())
    finite_values = tensor[finite]
    if finite_values.numel() > 0:
        finite_min = float(finite_values.min().item())
        finite_max = float(finite_values.max().item())
        range_text = f" finite range=[{finite_min:.6g}, {finite_max:.6g}]"
    else:
        range_text = " no finite values"
    raise ValueError(
        f"{name} contains {bad_count} non-finite values;{range_text}. "
        "This usually means the HDF5 conversion produced NaN/Inf values or the wrong data field was read."
    )


def _mat_field(path: Path, field: str) -> torch.Tensor:
    try:
        data = scipy.io.loadmat(path)[field]
    except NotImplementedError:
        with h5py.File(path) as handle:
            data = handle[field][()]
            data = np.transpose(data, axes=range(data.ndim - 1, -1, -1))
    return torch.tensor(data, dtype=torch.float32)


def _viscosity_value(viscosity_type: str) -> float:
    mapping = {"1e-3": 1e-3, "1e-4": 1e-4, "1e-5": 1e-5}
    if viscosity_type not in mapping:
        raise ValueError(f"Unsupported viscosity_type: {viscosity_type}")
    return mapping[viscosity_type]


def build_burgers_stage3_loaders(
    *,
    data_path: Path,
    batch_size: int = 64,
    sub: int = 32,
    ntrain: int = 1000,
    ntest: int = 200,
    num_workers: int = 0,
) -> LoaderBundle:
    data = scipy.io.loadmat(data_path)
    x_data = torch.tensor(data["a"][:, ::sub], dtype=torch.float32)
    y_data = torch.tensor(data["u"][:, ::sub], dtype=torch.float32)
    if x_data.shape[0] < ntrain + ntest:
        raise ValueError(f"Burgers file has {x_data.shape[0]} samples; need {ntrain + ntest}.")

    train_x = x_data[:ntrain].unsqueeze(-1)
    train_y = y_data[:ntrain].unsqueeze(-1)
    test_x = x_data[-ntest:].unsqueeze(-1)
    test_y = y_data[-ntest:].unsqueeze(-1)
    grid_size = train_x.shape[1]
    dx = 1.0 / max(grid_size - 1, 1)
    condition = [math.log10(10.0), float(dx), float(sub), 1.0]
    train_c = _constant_conditions(ntrain, condition)
    test_c = _constant_conditions(ntest, condition)

    return LoaderBundle(
        train_loader=DataLoader(
            TensorDataset(train_x, train_y, train_c),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        test_loader=DataLoader(
            TensorDataset(test_x, test_y, test_c),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        input_channels=1,
        output_channels=1,
        condition_dim=len(condition),
        t_out=1,
        spatial_dim=1,
        metadata={
            "dataset": "burgers",
            "condition_fields": ["log10_reynolds", "dx", "sub", "is_periodic"],
            "condition_values": condition,
            "train_shape": list(train_x.shape),
            "test_shape": list(test_x.shape),
        },
    )


def _load_ns_tensor(path: Path, viscosity_type: str, total: int) -> torch.Tensor:
    if viscosity_type in {"1e-3", "1e-4"}:
        with h5py.File(path) as handle:
            data = handle["u"][..., :total]
        return torch.tensor(data, dtype=torch.float32).permute(3, 1, 2, 0)
    data = _mat_field(path, "u")
    if data.shape[0] >= total and data.ndim == 4:
        return data[:total]
    if data.shape[-1] >= total:
        return data[..., :total].permute(3, 0, 1, 2)
    raise ValueError(f"Cannot infer NS layout for shape {tuple(data.shape)}.")


def build_navier_stokes_stage3_loaders(
    *,
    data_path: Path,
    viscosity_type: str,
    batch_size: int = 10,
    t_in: int = 10,
    t_out: int = 40,
    sub: int = 1,
    ntrain: int = 1000,
    ntest: int = 200,
    dt: float = 1.0,
    num_workers: int = 0,
    split: SampleSplit | None = None,
) -> LoaderBundle:
    if split is None:
        if ntrain != 1000 or ntest != 200:
            raise ValueError(
                "Changing NS ntrain/ntest without an explicit SampleSplit changes the historical test set. "
                "Pass split=SampleSplit(...) or use the default 1000/200 protocol."
            )
        train_indices = tuple(range(1000))
        test_indices = tuple(range(1000, 1200))
        split_name = "legacy_1000_200"
    else:
        split.validate(max_index((split.train, split.val, split.test)) + 1)
        train_indices = split.train
        test_indices = split.test
        split_name = split.name
        ntrain = len(train_indices)
        ntest = len(test_indices)
    needed = max_index((train_indices, test_indices)) + 1
    data = _load_ns_tensor(data_path, viscosity_type, needed)
    if data.shape[-1] < t_in + t_out:
        raise ValueError(f"NS data has {data.shape[-1]} time steps; need {t_in + t_out}.")

    train_index_tensor = torch.tensor(train_indices, dtype=torch.long)
    test_index_tensor = torch.tensor(test_indices, dtype=torch.long)
    train_x = data[train_index_tensor, ::sub, ::sub, :t_in]
    train_y = data[train_index_tensor, ::sub, ::sub, t_in : t_in + t_out]
    test_x = data[test_index_tensor, ::sub, ::sub, :t_in]
    test_y = data[test_index_tensor, ::sub, ::sub, t_in : t_in + t_out]
    viscosity = _viscosity_value(viscosity_type)
    dx = 1.0 / max(train_x.shape[1] - 1, 1)
    dy = 1.0 / max(train_x.shape[2] - 1, 1)
    condition = [math.log10(viscosity), dx, dy, dt, float(t_in), float(t_out), float(sub)]
    train_c = _constant_conditions(ntrain, condition)
    test_c = _constant_conditions(ntest, condition)

    return LoaderBundle(
        train_loader=DataLoader(
            TensorDataset(train_x, train_y, train_c),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        test_loader=DataLoader(
            TensorDataset(test_x, test_y, test_c),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        input_channels=t_in,
        output_channels=1,
        condition_dim=len(condition),
        t_out=t_out,
        spatial_dim=2,
        metadata={
            "dataset": f"navier_stokes_{viscosity_type}",
            "condition_fields": ["log10_viscosity", "dx", "dy", "dt", "t_in", "t_out", "sub"],
            "condition_values": condition,
            "split_name": split_name,
            "train_indices": list(train_indices),
            "test_indices": list(test_indices),
            "train_shape": list(train_x.shape),
            "test_shape": list(test_x.shape),
        },
    )


def _sample_to_xy_time(
    sample: np.ndarray,
    required_steps: int,
    sub: int,
    key: str,
) -> tuple[torch.Tensor, str]:
    if sample.ndim == 4:
        if sample.shape[-1] < 1:
            raise ValueError(f"{key} has no channel dimension: shape={sample.shape}.")
        if sample.shape[1] == sample.shape[2] and sample.shape[0] >= required_steps:
            sample = sample[:required_steps, ::sub, ::sub, 0]
            layout = "TXYC"
        elif sample.shape[0] == sample.shape[1] and sample.shape[2] >= required_steps:
            sample = sample[::sub, ::sub, :required_steps, 0]
            layout = "XYTC"
        else:
            raise ValueError(
                f"{key} must look like (T,X,Y,C) or (X,Y,T,C); got {sample.shape}."
            )
    elif sample.ndim == 3:
        if sample.shape[1] == sample.shape[2] and sample.shape[0] >= required_steps:
            sample = sample[:required_steps, ::sub, ::sub]
            layout = "TXY"
        elif sample.shape[0] == sample.shape[1] and sample.shape[2] >= required_steps:
            sample = sample[::sub, ::sub, :required_steps]
            layout = "XYT"
        else:
            raise ValueError(f"{key} must look like (T,X,Y) or (X,Y,T); got {sample.shape}.")
    else:
        raise ValueError(f"{key} must be 3D or 4D, got {sample.shape}.")

    if layout.startswith("T"):
        tensor = torch.tensor(sample, dtype=torch.float32).permute(1, 2, 0)
    else:
        tensor = torch.tensor(sample, dtype=torch.float32)
    return tensor, layout


def _load_root_shallow_water_dataset(
    dataset: h5py.Dataset,
    total: int,
    required_steps: int,
    sub: int,
) -> tuple[torch.Tensor, str]:
    if dataset.shape[0] < total:
        raise ValueError(f"/data has {dataset.shape[0]} samples; need {total}.")

    if dataset.ndim == 5 and dataset.shape[-1] >= 1:
        if dataset.shape[2] == dataset.shape[3] and dataset.shape[1] >= required_steps:
            data = dataset[:total, :required_steps, ::sub, ::sub, 0]
            return torch.tensor(data, dtype=torch.float32).permute(0, 2, 3, 1), "B-T-X-Y-C"
        if dataset.shape[1] == dataset.shape[2] and dataset.shape[3] >= required_steps:
            data = dataset[:total, ::sub, ::sub, :required_steps, 0]
            return torch.tensor(data, dtype=torch.float32), "B-X-Y-T-C"

    if dataset.ndim == 4:
        if dataset.shape[2] == dataset.shape[3] and dataset.shape[1] >= required_steps:
            data = dataset[:total, :required_steps, ::sub, ::sub]
            return torch.tensor(data, dtype=torch.float32).permute(0, 2, 3, 1), "B-T-X-Y"
        if dataset.shape[1] == dataset.shape[2] and dataset.shape[-1] >= required_steps:
            data = dataset[:total, ::sub, ::sub, :required_steps]
            return torch.tensor(data, dtype=torch.float32), "B-X-Y-T"

    raise ValueError(
        "Root /data must have shape (B,X,Y,T), (B,T,X,Y), (B,X,Y,T,C), or (B,T,X,Y,C); "
        f"got {dataset.shape}."
    )


def _load_shallow_water_data(path: Path, total: int, required_steps: int, sub: int) -> tuple[torch.Tensor, str]:
    with h5py.File(path) as handle:
        if "data" in handle and isinstance(handle["data"], h5py.Dataset):
            return _load_root_shallow_water_dataset(handle["data"], total, required_steps, sub)

        keys = sorted(k for k in handle.keys() if isinstance(handle[k], h5py.Group) and "data" in handle[k])
        if len(keys) < total:
            raise ValueError(f"Grouped file has {len(keys)} samples; need {total}.")
        first, layout = _sample_to_xy_time(
            handle[f"{keys[0]}/data"][:],
            required_steps,
            sub,
            f"{keys[0]}/data",
        )
        data = torch.empty((total, *first.shape), dtype=torch.float32)
        data[0] = first
        for index, key in enumerate(keys[1:total], start=1):
            sample, sample_layout = _sample_to_xy_time(
                handle[f"{key}/data"][:], required_steps, sub, f"{key}/data"
            )
            if sample_layout != layout:
                raise ValueError(f"Grouped shallow-water layouts differ: {layout} vs {sample_layout} at {key}.")
            data[index] = sample
        return data, f"grouped-{layout}"


def build_shallow_water_stage3_loaders(
    *,
    data_path: Path,
    batch_size: int = 5,
    t_in: int = 10,
    t_out: int = 40,
    sub: int = 1,
    ntrain: int = 900,
    ntest: int = 100,
    dt: float = 1.0,
    num_workers: int = 0,
) -> LoaderBundle:
    total = ntrain + ntest
    data, source_layout = _load_shallow_water_data(data_path, total, t_in + t_out, sub)
    _require_finite("shallow-water data", data)
    train_x = data[:ntrain, :, :, :t_in]
    train_y = data[:ntrain, :, :, t_in : t_in + t_out]
    test_x = data[-ntest:, :, :, :t_in]
    test_y = data[-ntest:, :, :, t_in : t_in + t_out]

    dx = 1.0 / max(train_x.shape[1] - 1, 1)
    dy = 1.0 / max(train_x.shape[2] - 1, 1)
    condition = [dx, dy, dt, float(t_in), float(t_out), float(sub), 1.0]
    train_c = _constant_conditions(ntrain, condition)
    test_c = _constant_conditions(ntest, condition)

    return LoaderBundle(
        train_loader=DataLoader(
            TensorDataset(train_x, train_y, train_c),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        test_loader=DataLoader(
            TensorDataset(test_x, test_y, test_c),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        input_channels=t_in,
        output_channels=1,
        condition_dim=len(condition),
        t_out=t_out,
        spatial_dim=2,
        metadata={
            "dataset": "shallow_water",
            "condition_fields": ["dx", "dy", "dt", "t_in", "t_out", "sub", "radial_dam_break_flag"],
            "condition_values": condition,
            "source_layout": source_layout,
            "train_shape": list(train_x.shape),
            "test_shape": list(test_x.shape),
        },
    )
