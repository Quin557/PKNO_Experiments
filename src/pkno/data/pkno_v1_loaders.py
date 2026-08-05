"""Dedicated data loaders for Stage3_2 PKNO_v1.

The historical NS train/test trajectories remain unchanged.  Unlike the old
``ntrain + ntest`` loader, every V1 loader receives explicit indices, so a
training-scale experiment cannot silently change the test trajectories.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, Sampler, TensorDataset

from pkno.data import stage3_loaders as legacy
from pkno.data.split_protocol import SampleSplit, final_split, legacy_ns_split, max_index, tuning_split


@dataclass(frozen=True)
class V1LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader | None
    test_loader: DataLoader
    input_channels: int
    output_channels: int
    condition_dim: int
    t_out: int
    spatial_dim: int
    metadata: dict[str, Any]


class BalancedTwoConditionBatchSampler(Sampler[list[int]]):
    """Make every joint-NS training batch contain equal viscosities."""

    def __init__(self, first_count: int, second_count: int, batch_size: int, seed: int = 42) -> None:
        if batch_size <= 1 or batch_size % 2:
            raise ValueError("Joint NS batch_size must be a positive even number.")
        if first_count != second_count:
            raise ValueError("Joint NS requires equal trajectory counts per condition.")
        self.first_count = first_count
        self.second_count = second_count
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        first = torch.randperm(self.first_count, generator=generator).tolist()
        second = (torch.randperm(self.second_count, generator=generator) + self.first_count).tolist()
        half = self.batch_size // 2
        for start in range(0, self.first_count - half + 1, half):
            yield first[start : start + half] + second[start : start + half]

    def __len__(self) -> int:
        return self.first_count // (self.batch_size // 2)


def _loader(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, *, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(TensorDataset(x, y, c), batch_size=batch_size, shuffle=shuffle, num_workers=workers)


def _empty_or_loader(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, *, batch_size: int, workers: int) -> DataLoader | None:
    if x.shape[0] == 0:
        return None
    return _loader(x, y, c, batch_size=batch_size, shuffle=False, workers=workers)


def _select(data: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    if not indices:
        return data[:0]
    return data[torch.tensor(indices, dtype=torch.long)]


def _conditions(count: int, values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, -1).repeat(count, 1)


def _bundle_from_data(
    *,
    data: torch.Tensor,
    split: SampleSplit,
    t_in: int,
    t_out: int,
    condition: list[float],
    condition_fields: list[str],
    batch_size: int,
    workers: int,
    spatial_dim: int,
    metadata: dict[str, Any],
) -> V1LoaderBundle:
    split.validate(data.shape[0])
    if data.shape[-1] < t_in + t_out:
        raise ValueError(f"Data has {data.shape[-1]} time frames; need {t_in + t_out}.")

    def xy(indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected = _select(data, indices)
        return (
            selected[..., :t_in],
            selected[..., t_in : t_in + t_out],
            _conditions(selected.shape[0], condition),
        )

    train_x, train_y, train_c = xy(split.train)
    val_x, val_y, val_c = xy(split.val)
    test_x, test_y, test_c = xy(split.test)
    metadata.update(
        {
            "split_name": split.name,
            "split_indices": {"train": list(split.train), "val": list(split.val), "test": list(split.test)},
            "condition_fields": condition_fields,
            "condition_values": condition,
            "train_shape": list(train_x.shape),
            "val_shape": list(val_x.shape),
            "test_shape": list(test_x.shape),
        }
    )
    return V1LoaderBundle(
        train_loader=_loader(train_x, train_y, train_c, batch_size=batch_size, shuffle=True, workers=workers),
        val_loader=_empty_or_loader(val_x, val_y, val_c, batch_size=batch_size, workers=workers),
        test_loader=_loader(test_x, test_y, test_c, batch_size=batch_size, shuffle=False, workers=workers),
        input_channels=t_in,
        output_channels=1,
        condition_dim=len(condition),
        t_out=t_out,
        spatial_dim=spatial_dim,
        metadata=metadata,
    )


def build_ns_v1_loaders(
    *, data_path: Path, viscosity_type: str, batch_size: int = 10, t_in: int = 10,
    t_out: int = 40, sub: int = 1, dt: float = 1.0, split: SampleSplit | None = None,
    num_workers: int = 0,
) -> V1LoaderBundle:
    split = split or legacy_ns_split()
    needed = max_index((split.train, split.val, split.test)) + 1
    if viscosity_type not in {"1e-3", "1e-4"}:
        raise ValueError("PKNO_v1 main protocol supports NS viscosities 1e-3 and 1e-4 only.")
    with h5py.File(data_path) as handle:
        raw = handle["u"][..., :needed]
    data = torch.tensor(raw, dtype=torch.float32).permute(3, 1, 2, 0)[:, ::sub, ::sub]
    if not torch.isfinite(data).all():
        raise ValueError("Navier-Stokes data contains NaN/Inf values.")
    viscosity = {"1e-3": 1e-3, "1e-4": 1e-4}[viscosity_type]
    dx = 1.0 / max(data.shape[1] - 1, 1)
    dy = 1.0 / max(data.shape[2] - 1, 1)
    return _bundle_from_data(
        data=data, split=split, t_in=t_in, t_out=t_out,
        condition=[math.log10(viscosity), dx, dy, dt, float(sub)],
        condition_fields=["log10_viscosity", "dx", "dy", "dt", "sub"],
        batch_size=batch_size, workers=num_workers, spatial_dim=2,
        metadata={"dataset": f"navier_stokes_{viscosity_type}", "source_path": str(data_path)},
    )


def build_burgers_v1_loaders(
    *, data_path: Path, batch_size: int = 64, sub: int = 32, split_mode: str = "final", num_workers: int = 0,
) -> V1LoaderBundle:
    split = (
        final_split(train_count=1000, test_count=200, name="pkno_v1_burgers_final_1000_200")
        if split_mode == "final"
        else tuning_split(train_count=900, val_count=100, test_count=200, name="pkno_v1_burgers_tuning_900_100_200")
    )
    raw = scipy.io.loadmat(data_path)
    x = torch.tensor(raw["a"][:, ::sub], dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(raw["u"][:, ::sub], dtype=torch.float32).unsqueeze(-1)
    split.validate(x.shape[0])
    condition = [math.log10(10.0), 1.0 / max(x.shape[1] - 1, 1), float(sub), 1.0]

    def select(indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _select(x, indices), _select(y, indices), _conditions(len(indices), condition)

    train_x, train_y, train_c = select(split.train)
    val_x, val_y, val_c = select(split.val)
    test_x, test_y, test_c = select(split.test)
    return V1LoaderBundle(
        train_loader=_loader(train_x, train_y, train_c, batch_size=batch_size, shuffle=True, workers=num_workers),
        val_loader=_empty_or_loader(val_x, val_y, val_c, batch_size=batch_size, workers=num_workers),
        test_loader=_loader(test_x, test_y, test_c, batch_size=batch_size, shuffle=False, workers=num_workers),
        input_channels=1, output_channels=1, condition_dim=len(condition), t_out=1, spatial_dim=1,
        metadata={"dataset": "burgers", "split_name": split.name, "split_indices": {"train": list(split.train), "val": list(split.val), "test": list(split.test)}, "condition_fields": ["log10_reynolds", "dx", "sub", "is_periodic"]},
    )


def build_shallow_water_v1_loaders(
    *, data_path: Path, batch_size: int = 5, t_in: int = 10, t_out: int = 40,
    sub: int = 1, dt: float = 0.01, split_mode: str = "final", num_workers: int = 0,
) -> V1LoaderBundle:
    split = (
        final_split(train_count=900, test_count=100, name="pkno_v1_shallow_final_900_100")
        if split_mode == "final"
        else tuning_split(train_count=800, val_count=100, test_count=100, name="pkno_v1_shallow_tuning_800_100_100")
    )
    needed = max_index((split.train, split.val, split.test)) + 1
    data, layout = legacy._load_shallow_water_data(data_path, needed, t_in + t_out, sub)
    legacy._require_finite("shallow-water data", data)
    dx = 1.0 / max(data.shape[1] - 1, 1)
    dy = 1.0 / max(data.shape[2] - 1, 1)
    return _bundle_from_data(
        data=data, split=split, t_in=t_in, t_out=t_out,
        condition=[dx, dy, dt, float(sub), 1.0],
        condition_fields=["dx", "dy", "dt", "sub", "radial_dam_break_flag"],
        batch_size=batch_size, workers=num_workers, spatial_dim=2,
        metadata={"dataset": "shallow_water", "source_path": str(data_path), "source_layout": layout},
    )


def build_joint_ns_v1_loaders(
    *, data_v1e3: Path, data_v1e4: Path, batch_size: int = 10, t_in: int = 10,
    t_out: int = 40, sub: int = 1, dt: float = 1.0, num_workers: int = 0, seed: int = 42,
) -> V1LoaderBundle:
    first = build_ns_v1_loaders(data_path=data_v1e3, viscosity_type="1e-3", batch_size=batch_size, t_in=t_in, t_out=t_out, sub=sub, dt=dt, num_workers=num_workers)
    second = build_ns_v1_loaders(data_path=data_v1e4, viscosity_type="1e-4", batch_size=batch_size, t_in=t_in, t_out=t_out, sub=sub, dt=dt, num_workers=num_workers)

    def tensors(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dataset = loader.dataset
        if not isinstance(dataset, TensorDataset):
            raise TypeError("PKNO_v1 joint loader requires TensorDataset components.")
        return dataset.tensors  # type: ignore[return-value]

    first_train = tensors(first.train_loader)
    second_train = tensors(second.train_loader)
    train_dataset = TensorDataset(*(torch.cat([a, b], dim=0) for a, b in zip(first_train, second_train)))
    sampler = BalancedTwoConditionBatchSampler(first_train[0].shape[0], second_train[0].shape[0], batch_size=batch_size, seed=seed)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=num_workers)

    def merged_eval(which: str) -> DataLoader:
        left = getattr(first, f"{which}_loader")
        right = getattr(second, f"{which}_loader")
        if left is None or right is None:
            raise ValueError("Joint NS requires validation and test data for both viscosities.")
        return DataLoader(TensorDataset(*(torch.cat([a, b], dim=0) for a, b in zip(tensors(left), tensors(right)))), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return V1LoaderBundle(
        train_loader=train_loader, val_loader=merged_eval("val"), test_loader=merged_eval("test"),
        input_channels=first.input_channels, output_channels=1, condition_dim=first.condition_dim,
        t_out=t_out, spatial_dim=2,
        metadata={"dataset": "navier_stokes_joint_v1e3_v1e4", "balanced_sampling": True, "per_condition_train": 1000, "condition_fields": first.metadata["condition_fields"], "first": first.metadata, "second": second.metadata},
    )
