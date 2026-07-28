from __future__ import annotations

import torch

from pkno.operators.koopman_parameterized import ConditionEncoder


def test_condition_encoder_rejects_wrong_condition_dim():
    encoder = ConditionEncoder(condition_dim=3, state_embed_dim=4, output_dim=8)
    condition = torch.randn(2, 2)
    state = torch.randn(2, 4)
    try:
        encoder(condition, state)
    except ValueError as exc:
        assert "Expected condition dim" in str(exc)
    else:
        raise AssertionError("ConditionEncoder should reject wrong condition dim.")


def test_condition_encoder_output_shape():
    encoder = ConditionEncoder(condition_dim=3, state_embed_dim=4, output_dim=8)
    out = encoder(torch.randn(2, 3), torch.randn(2, 4))
    assert out.shape == (2, 8)
