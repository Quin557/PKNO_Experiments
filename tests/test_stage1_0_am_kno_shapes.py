from __future__ import annotations

import torch

from amkno.models import AMKNO1d, AMKNO2d


def test_am_kno_1d_freq_only_forward_shape():
    model = AMKNO1d(
        input_channels=1,
        output_channels=1,
        observable_dim=8,
        decompose=2,
        max_modes=0,
        frequency_basis_dim=4,
        condition_mode="freq",
        generator_hidden_dim=16,
        generator_depth=1,
    )
    x = torch.randn(2, 32, 1)
    condition = torch.randn(2, 4)
    pred, recon = model(x, condition)
    assert pred.shape == (2, 32, 1)
    assert recon.shape == x.shape


def test_am_kno_2d_state_conditioned_forward_shape():
    model = AMKNO2d(
        input_channels=10,
        output_channels=1,
        condition_dim=7,
        observable_dim=8,
        decompose=2,
        max_modes=4,
        frequency_basis_dim=4,
        condition_mode="state",
        state_embed_dim=8,
        condition_embed_dim=16,
        generator_hidden_dim=16,
        generator_depth=1,
        operator_factorization="full",
    )
    x = torch.randn(2, 16, 16, 10)
    condition = torch.randn(2, 7)
    pred, recon = model(x, condition)
    assert pred.shape == (2, 16, 16, 1)
    assert recon.shape == x.shape


def test_am_kno_2d_state_static_forward_shape():
    model = AMKNO2d(
        input_channels=10,
        output_channels=1,
        condition_dim=7,
        observable_dim=8,
        decompose=1,
        max_modes=3,
        frequency_basis_dim=4,
        condition_mode="state_static",
        state_embed_dim=8,
        condition_embed_dim=16,
        generator_hidden_dim=16,
        generator_depth=1,
        operator_factorization="full",
    )
    x = torch.randn(2, 8, 8, 10)
    condition = torch.randn(2, 7)
    pred, recon = model(x, condition)
    assert pred.shape == (2, 8, 8, 1)
    assert recon.shape == x.shape


def test_am_kno_2d_default_factorized_freq_forward_shape():
    model = AMKNO2d(
        input_channels=10,
        output_channels=1,
        condition_dim=7,
        observable_dim=8,
        decompose=1,
        max_modes=3,
        frequency_basis_dim=4,
        condition_mode="freq",
        generator_hidden_dim=16,
        generator_depth=1,
        operator_factorization="factorized",
        factorized_rank=1,
    )
    x = torch.randn(2, 8, 8, 10)
    condition = torch.randn(2, 7)
    pred, recon = model(x, condition)
    assert pred.shape == (2, 8, 8, 1)
    assert recon.shape == x.shape
