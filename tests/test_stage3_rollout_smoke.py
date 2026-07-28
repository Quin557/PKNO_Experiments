from __future__ import annotations

import torch
from torch import nn

from pkno.trainers.train_rollout import RelativeL2, rollout_model


class TinyModel(nn.Module):
    output_channels = 1

    def forward(self, history: torch.Tensor, condition: torch.Tensor):
        del condition
        pred = history[..., -1:]
        recon = history
        return pred, recon


def test_rollout_model_scalar_history_update():
    model = TinyModel()
    history = torch.randn(2, 8, 3)
    target = torch.randn(2, 8, 2)
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
    assert out["pred_mse"].ndim == 0
