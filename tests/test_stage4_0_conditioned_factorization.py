from __future__ import annotations

import torch

from ampkno.frequency import ConditionedFactorizedMatrixGenerator2D, ConditionedMatrixGenerator
from ampkno.operators import AMParamKoopmanOperator2D


def test_conditioned_matrix_generator_shape():
    generator = ConditionedMatrixGenerator(
        freq_embed_dim=5,
        condition_embed_dim=6,
        observable_dim=4,
        hidden_dim=8,
        depth=1,
    )
    freq_embed = torch.randn(3, 2, 5)
    condition = torch.randn(2, 6)
    weights = generator(freq_embed, condition)
    assert weights.shape == (2, 3, 2, 4, 4)
    assert weights.is_complex()


def test_conditioned_factorized_generator_2d_shapes():
    generator = ConditionedFactorizedMatrixGenerator2D(
        freq_embed_dim=5,
        condition_embed_dim=6,
        observable_dim=4,
        rank=2,
        hidden_dim=8,
        depth=1,
    )
    freq_x = torch.randn(7, 5)
    freq_y = torch.randn(4, 5)
    condition = torch.randn(2, 6)
    factor_x, factor_y = generator(freq_x, freq_y, condition)
    assert factor_x.shape == (2, 7, 4, 4, 2)
    assert factor_y.shape == (2, 4, 4, 4, 2)
    assert factor_x.is_complex()
    assert factor_y.is_complex()


def test_am_param_koopman_2d_factorized_forward_shape():
    op = AMParamKoopmanOperator2D(
        observable_dim=4,
        max_modes=3,
        frequency_basis_dim=4,
        condition_embed_dim=6,
        generator_hidden_dim=8,
        generator_depth=1,
        operator_factorization="factorized",
        factorized_rank=2,
    )
    x = torch.randn(2, 4, 8, 8)
    condition = torch.randn(2, 6)
    y = op(x, condition)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_memory_efficient_factorized_march_matches_direct_einsum():
    x_ft = torch.randn(2, 3, 4, 5, dtype=torch.cfloat)
    factor_x = torch.randn(2, 4, 3, 6, 2, dtype=torch.cfloat)
    factor_y = torch.randn(2, 5, 3, 6, 2, dtype=torch.cfloat)
    expected = torch.einsum("bixy,bxior,byior->boxy", x_ft, factor_x, factor_y)
    actual = AMParamKoopmanOperator2D._time_march_factorized(x_ft, (factor_x, factor_y), input_chunk_size=2)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_am_param_koopman_2d_full_forward_shape():
    op = AMParamKoopmanOperator2D(
        observable_dim=4,
        max_modes=3,
        frequency_basis_dim=4,
        condition_embed_dim=6,
        generator_hidden_dim=8,
        generator_depth=1,
        operator_factorization="full",
    )
    x = torch.randn(2, 4, 8, 8)
    condition = torch.randn(2, 6)
    y = op(x, condition)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
