from __future__ import annotations

import torch

from pkno_u.stability import ContractiveTransition


def test_contractive_transition_bounds_every_complex_matrix():
    transition = ContractiveTransition(max_norm=0.7)
    matrices = torch.randn(2, 4, 5, 5, dtype=torch.cfloat) * 10.0
    projected = transition(matrices)
    spectral = torch.linalg.matrix_norm(projected, ord=2)
    assert torch.all(spectral <= 0.700001)
