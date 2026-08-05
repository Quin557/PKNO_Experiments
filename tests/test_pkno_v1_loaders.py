from __future__ import annotations

import pytest

from pkno.data.pkno_v1_loaders import BalancedTwoConditionBatchSampler
from pkno.data.split_protocol import SampleSplit, legacy_ns_split


def test_legacy_ns_split_preserves_historical_indices() -> None:
    split = legacy_ns_split()
    assert split.train[0] == 0 and split.train[-1] == 999
    assert split.test[0] == 1000 and split.test[-1] == 1199
    assert split.val[0] == 1200 and split.val[-1] == 1399
    split.validate(5000)


def test_split_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        SampleSplit(train=(0, 1), val=(1,), test=(2,), name="bad").validate(3)


def test_joint_sampler_balances_each_batch() -> None:
    sampler = BalancedTwoConditionBatchSampler(10, 10, batch_size=4)
    for batch in sampler:
        assert sum(index < 10 for index in batch) == 2
        assert sum(index >= 10 for index in batch) == 2
