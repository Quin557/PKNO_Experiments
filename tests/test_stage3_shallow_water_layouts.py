from __future__ import annotations

import h5py
import numpy as np

from pkno.data.stage3_loaders import _load_shallow_water_data


def test_root_shallow_water_bxytc_layout(tmp_path):
    path = tmp_path / "converted_swe.h5"
    raw = np.arange(2 * 4 * 4 * 5, dtype=np.float32).reshape(2, 4, 4, 5, 1)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=raw)

    data, layout = _load_shallow_water_data(path, total=2, required_steps=3, sub=1)

    assert layout == "B-X-Y-T-C"
    assert data.shape == (2, 4, 4, 3)
    np.testing.assert_allclose(data.numpy(), raw[..., :3, 0])


def test_grouped_shallow_water_txyc_layout(tmp_path):
    path = tmp_path / "grouped_swe.h5"
    raw = np.arange(2 * 5 * 4 * 4, dtype=np.float32).reshape(2, 5, 4, 4, 1)
    with h5py.File(path, "w") as handle:
        for index in range(2):
            group = handle.create_group(f"{index:04d}")
            group.create_dataset("data", data=raw[index])

    data, layout = _load_shallow_water_data(path, total=2, required_steps=3, sub=1)

    assert layout == "grouped-TXYC"
    assert data.shape == (2, 4, 4, 3)
    np.testing.assert_allclose(data.numpy(), raw[:, :3, :, :, 0].transpose(0, 2, 3, 1))
