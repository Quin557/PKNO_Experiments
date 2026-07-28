from __future__ import annotations

import torch

from ampkno.models import AMPKNO1d, AMPKNO2d


def test_am_pkno_1d_forward_shape():
    model = AMPKNO1d(
        input_channels=1,
        output_channels=1,
        condition_dim=4,
        observable_dim=12,
        decompose=2,
        max_modes=4,
        frequency_basis_dim=4,
        dictionary_hidden_dim=16,
        dictionary_depth=1,
        basis_kind="burgers",
        decoder_hidden_dim=16,
        condition_embed_dim=8,
        state_embed_dim=8,
        generator_hidden_dim=16,
        generator_depth=1,
    )
    history = torch.randn(2, 32, 1)
    condition = torch.randn(2, 4)
    pred, recon = model(history, condition)
    assert pred.shape == (2, 32, 1)
    assert recon.shape == history.shape
    assert torch.isfinite(pred).all()


def test_am_pkno_2d_factorized_forward_shape():
    model = AMPKNO2d(
        input_channels=10,
        output_channels=1,
        condition_dim=7,
        observable_dim=12,
        decompose=2,
        max_modes=4,
        frequency_basis_dim=4,
        dictionary_hidden_dim=16,
        dictionary_depth=1,
        basis_kind="navier_stokes",
        decoder_hidden_dim=16,
        condition_embed_dim=8,
        state_embed_dim=8,
        generator_hidden_dim=16,
        generator_depth=1,
        operator_factorization="factorized",
        factorized_rank=1,
    )
    history = torch.randn(2, 16, 16, 10)
    condition = torch.randn(2, 7)
    pred, recon = model(history, condition)
    assert pred.shape == (2, 16, 16, 1)
    assert recon.shape == history.shape
    assert torch.isfinite(pred).all()


def test_am_pkno_2d_full_forward_shape():
    model = AMPKNO2d(
        input_channels=10,
        output_channels=1,
        condition_dim=7,
        observable_dim=12,
        decompose=2,
        max_modes=4,
        frequency_basis_dim=4,
        dictionary_hidden_dim=16,
        dictionary_depth=1,
        basis_kind="shallow_water",
        decoder_hidden_dim=16,
        condition_embed_dim=8,
        state_embed_dim=8,
        generator_hidden_dim=16,
        generator_depth=1,
        operator_factorization="full",
    )
    history = torch.randn(2, 16, 16, 10)
    condition = torch.randn(2, 7)
    pred, recon = model(history, condition)
    assert pred.shape == (2, 16, 16, 1)
    assert recon.shape == history.shape
    assert torch.isfinite(pred).all()
