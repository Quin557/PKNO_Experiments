from __future__ import annotations

import torch
from torch import nn

from amkno.models import AMKNO1d
from pkno.trainers.train_rollout import RelativeL2, rollout_model


def test_am_kno_rollout_smoke():
    model = AMKNO1d(
        input_channels=3,
        output_channels=1,
        observable_dim=8,
        decompose=1,
        max_modes=4,
        frequency_basis_dim=4,
        condition_mode="freq",
        generator_hidden_dim=16,
        generator_depth=1,
    )
    history = torch.randn(2, 16, 3)
    target = torch.randn(2, 16, 2)
    condition = torch.randn(2, 4)
    out = rollout_model(
        model,
        history,
        target,
        condition,
        t_out=2,
        output_channels=1,
        mse=nn.MSELoss(),
        rel_l2=RelativeL2(),
    )
    assert out["pred"].shape == target.shape
    assert torch.isfinite(out["pred_mse"])
