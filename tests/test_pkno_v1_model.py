from __future__ import annotations

import torch

from pkno.models.pkno_v1 import PKNOV11d, PKNOV12d


def test_pkno_v1_1d_residual_and_shapes() -> None:
    model = PKNOV11d(input_channels=1, output_channels=1, condition_dim=4, observable_dim=12, modes=4, decompose=2, dictionary_hidden_dim=16, condition_embed_dim=16, state_embed_dim=8, koopman_hidden_dim=16)
    x = torch.randn(2, 32, 1)
    c = torch.randn(2, 4)
    output = model(x, c)
    assert output.prediction.shape == x.shape
    assert output.reconstruction.shape == x.shape
    assert output.state_correction.shape == (2, 16)
    assert torch.all(output.gate <= 0.1)


def test_pkno_v1_2d_shapes() -> None:
    model = PKNOV12d(input_channels=10, output_channels=1, condition_dim=5, observable_dim=12, modes=4, decompose=2, dictionary_hidden_dim=16, condition_embed_dim=16, state_embed_dim=8, koopman_hidden_dim=16)
    output = model(torch.randn(2, 16, 16, 10), torch.randn(2, 5))
    assert output.prediction.shape == (2, 16, 16, 1)
    assert output.reconstruction.shape == (2, 16, 16, 10)


def test_pkno_v1_zero_delta_is_identity_residual() -> None:
    model = PKNOV11d(input_channels=1, output_channels=1, condition_dim=4, observable_dim=12, modes=4, decompose=1, dictionary_hidden_dim=16, condition_embed_dim=16, state_embed_dim=8, koopman_hidden_dim=16)
    for parameter in model.pred_decoder.parameters():
        parameter.data.zero_()
    x = torch.randn(2, 32, 1)
    assert torch.allclose(model(x, torch.randn(2, 4)).prediction, x)
