from __future__ import annotations

import torch

from pkno_u.unet import high_pass


def test_high_pass_removes_1d_constant_component():
    residual = torch.ones(2, 3, 32)
    filtered = high_pass(residual, cutoff=0.5)
    assert torch.allclose(filtered, torch.zeros_like(filtered), atol=1e-6)


def test_high_pass_keeps_2d_checkerboard_component():
    axis = torch.arange(16)
    checkerboard = ((axis[:, None] + axis[None, :]) % 2).float() * 2.0 - 1.0
    residual = checkerboard.reshape(1, 1, 16, 16)
    filtered = high_pass(residual, cutoff=0.5)
    assert torch.linalg.vector_norm(filtered) > 0.9 * torch.linalg.vector_norm(residual)
