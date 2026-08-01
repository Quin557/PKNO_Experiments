from __future__ import annotations

import torch

from pkno_u.conditioning import ConditioningModule


def test_physical_only_is_independent_of_history():
    module = ConditioningModule(condition_dim=3, input_channels=2, output_dim=8, mode="physical_only")
    condition = torch.randn(2, 3)
    first, first_gate = module(condition, torch.randn(2, 8, 8, 2))
    second, second_gate = module(condition, torch.randn(2, 8, 8, 2))
    assert torch.allclose(first, second)
    assert torch.allclose(first_gate, torch.zeros_like(first_gate))
    assert torch.allclose(second_gate, torch.zeros_like(second_gate))


def test_gated_state_condition_is_bounded():
    module = ConditioningModule(condition_dim=3, input_channels=2, output_dim=8, mode="physical_gated_state")
    _, gate = module(torch.randn(2, 3), torch.randn(2, 8, 8, 2))
    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)
