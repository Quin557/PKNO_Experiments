from __future__ import annotations

import torch

from pkno.models.param_kno import ParamKNO1d, ParamKNO2d


def test_param_kno_1d_forward_shape():
    model = ParamKNO1d(
        input_channels=1,
        output_channels=1,
        condition_dim=4,
        observable_dim=8,
        modes=4,
        decompose=2,
        condition_embed_dim=16,
        state_embed_dim=8,
        koopman_hidden_dim=16,
        dictionary_hidden_dim=16,
    )
    x = torch.randn(2, 32, 1)
    c = torch.randn(2, 4)
    pred, recon = model(x, c)
    assert pred.shape == (2, 32, 1)
    assert recon.shape == x.shape


def test_param_kno_2d_forward_shape():
    model = ParamKNO2d(
        input_channels=10,
        output_channels=1,
        condition_dim=7,
        observable_dim=8,
        modes=4,
        decompose=2,
        condition_embed_dim=16,
        state_embed_dim=8,
        koopman_hidden_dim=16,
        dictionary_hidden_dim=16,
    )
    x = torch.randn(2, 16, 16, 10)
    c = torch.randn(2, 7)
    pred, recon = model(x, c)
    assert pred.shape == (2, 16, 16, 1)
    assert recon.shape == x.shape
