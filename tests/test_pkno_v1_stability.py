from __future__ import annotations

import torch

from pkno.losses.pkno_v1_stability import growth_envelope_penalty, state_correction_penalty, temporal_state_penalty
from pkno.trainers.train_pkno_v1 import curriculum_horizon


def test_stability_losses_are_finite_and_soft() -> None:
    correction = torch.randn(3, 8)
    assert state_correction_penalty(correction).item() > 0
    assert temporal_state_penalty(correction, correction.detach()).item() == 0
    current = torch.ones(2, 4, 4, 1)
    assert growth_envelope_penalty(current, current, 1.1).item() == 0
    assert growth_envelope_penalty(2 * current, current, 1.1).item() > 0


def test_curriculum_reaches_full_rollout() -> None:
    from pkno.trainers.train_pkno_v1 import PKNOV1TrainConfig
    config = PKNOV1TrainConfig()
    assert curriculum_horizon(0, 40, config) == (1, True)
    assert curriculum_horizon(60, 40, config) == (5, False)
    assert curriculum_horizon(90, 40, config) == (10, False)
    assert curriculum_horizon(100, 40, config) == (40, False)
