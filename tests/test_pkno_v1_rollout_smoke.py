from __future__ import annotations

import torch
from torch import nn

from pkno.models.pkno_v1 import PKNOV1Output
from pkno.trainers.train_pkno_v1 import rollout_v1


class TinyV1(nn.Module):
    def forward(self, history: torch.Tensor, condition: torch.Tensor) -> PKNOV1Output:
        del condition
        return PKNOV1Output(history[..., -1:], history, torch.zeros(history.shape[0], 2), torch.zeros(history.shape[0], 2))


def test_v1_rollout_smoke() -> None:
    history = torch.randn(2, 8, 3)
    target = torch.randn(2, 8, 4)
    output = rollout_v1(TinyV1(), history, target, torch.zeros(2, 4), horizon=4, teacher_forced=False, growth_ceiling=1.2)
    assert output["prediction"].shape == target.shape
    assert torch.isfinite(output["pred_mse"])
