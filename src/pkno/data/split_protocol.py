"""Explicit, validated sample splits for reproducible PDE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SampleSplit:
    """Non-overlapping trajectory indices for one dataset."""

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]
    name: str

    def validate(self, total_samples: int) -> None:
        groups = {"train": self.train, "val": self.val, "test": self.test}
        for group_name, indices in groups.items():
            if len(indices) != len(set(indices)):
                raise ValueError(f"{self.name}: duplicate {group_name} indices.")
            if any(index < 0 or index >= total_samples for index in indices):
                raise ValueError(
                    f"{self.name}: {group_name} indices must be in [0, {total_samples - 1}]."
                )
        seen: set[int] = set()
        for group_name, indices in groups.items():
            overlap = seen.intersection(indices)
            if overlap:
                raise ValueError(f"{self.name}: {group_name} overlaps a prior split at {sorted(overlap)[:5]}.")
            seen.update(indices)


def contiguous(start: int, count: int) -> tuple[int, ...]:
    if start < 0 or count < 0:
        raise ValueError("start and count must be non-negative.")
    return tuple(range(start, start + count))


def legacy_ns_split(*, train_count: int = 1000, test_count: int = 200) -> SampleSplit:
    """Historical Stage 0/3 NS protocol with a separate unused validation range."""

    if train_count != 1000 or test_count != 200:
        raise ValueError(
            "The legacy NS protocol is defined only for ntrain=1000 and ntest=200. "
            "Create an explicit SampleSplit before changing data scale."
        )
    return SampleSplit(
        train=contiguous(0, 1000),
        test=contiguous(1000, 200),
        val=contiguous(1200, 200),
        name="pkno_v1_ns_legacy_1000_200",
    )


def tuning_split(*, train_count: int, val_count: int, test_count: int, name: str) -> SampleSplit:
    return SampleSplit(
        train=contiguous(0, train_count),
        val=contiguous(train_count, val_count),
        test=contiguous(train_count + val_count, test_count),
        name=name,
    )


def final_split(*, train_count: int, test_count: int, name: str) -> SampleSplit:
    """Final retraining split. Validation is intentionally empty."""

    return SampleSplit(
        train=contiguous(0, train_count),
        val=(),
        test=contiguous(train_count, test_count),
        name=name,
    )


def max_index(groups: Iterable[tuple[int, ...]]) -> int:
    values = [index for group in groups for index in group]
    return max(values, default=-1)
