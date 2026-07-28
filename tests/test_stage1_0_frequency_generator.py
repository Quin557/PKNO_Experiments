from __future__ import annotations

import torch

from amkno.frequency import AmortizedMatrixGenerator, ChebyshevFrequencyEmbedding
from amkno.operators import AMKoopmanOperator1D, AMKoopmanOperator2D


def test_chebyshev_frequency_embedding_shape():
    embedding = ChebyshevFrequencyEmbedding(spatial_dim=2, basis_dim=4)
    freq = torch.tensor([[0.0, 0.5], [-0.5, 0.25]])
    out = embedding(freq)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_amortized_matrix_generator_freq_only_shape():
    generator = AmortizedMatrixGenerator(
        freq_embed_dim=5,
        observable_dim=4,
        hidden_dim=8,
        depth=1,
    )
    freq_embed = torch.randn(3, 5)
    weights = generator(freq_embed)
    assert weights.shape == (3, 4, 4)
    assert weights.is_complex()


def test_amortized_matrix_generator_state_conditioned_shape():
    generator = AmortizedMatrixGenerator(
        freq_embed_dim=5,
        observable_dim=4,
        condition_dim=6,
        hidden_dim=8,
        depth=1,
    )
    freq_embed = torch.randn(3, 5)
    condition = torch.randn(2, 6)
    weights = generator(freq_embed, condition)
    assert weights.shape == (2, 3, 4, 4)
    assert weights.is_complex()


def test_am_koopman_1d_forward_shape_all_modes():
    op = AMKoopmanOperator1D(
        observable_dim=4,
        max_modes=0,
        frequency_basis_dim=4,
        generator_hidden_dim=8,
        generator_depth=1,
    )
    x = torch.randn(2, 4, 16)
    y = op(x)
    assert y.shape == x.shape


def test_am_koopman_2d_forward_shape_capped_modes():
    op = AMKoopmanOperator2D(
        observable_dim=4,
        max_modes=3,
        frequency_basis_dim=4,
        generator_hidden_dim=8,
        generator_depth=1,
    )
    x = torch.randn(2, 4, 8, 8)
    y = op(x)
    assert y.shape == x.shape
