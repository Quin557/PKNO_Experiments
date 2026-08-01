from __future__ import annotations

import pytest
import torch

from pkno_u.models import PKNOU1d, PKNOU2d


@pytest.mark.parametrize("condition_mode", ["physical_only", "physical_compact_state", "physical_gated_state"])
def test_pkno_u_1d_shapes_for_all_condition_modes(condition_mode: str):
    model = PKNOU1d(
        input_channels=1,
        output_channels=1,
        condition_dim=4,
        observable_dim=12,
        modes=4,
        decompose=2,
        condition_mode=condition_mode,
        condition_embed_dim=16,
        state_embed_dim=4,
        koopman_hidden_dim=16,
        dictionary_hidden_dim=16,
        unet_base_channels=8,
    )
    prediction, reconstruction = model(torch.randn(2, 16, 1), torch.randn(2, 4))
    assert prediction.shape == (2, 16, 1)
    assert reconstruction.shape == (2, 16, 1)
    assert torch.isfinite(prediction).all()
    assert set(model.last_diagnostics) == {
        "condition_gate", "matrix_spectral_mean", "matrix_spectral_max", "latent_rms", "unet_highpass_rms"
    }


def test_pkno_u_2d_shape():
    model = PKNOU2d(
        input_channels=3,
        output_channels=1,
        condition_dim=7,
        observable_dim=12,
        modes=4,
        decompose=2,
        condition_embed_dim=16,
        koopman_hidden_dim=16,
        dictionary_hidden_dim=16,
        unet_base_channels=8,
    )
    prediction, reconstruction = model(torch.randn(2, 16, 16, 3), torch.randn(2, 7))
    assert prediction.shape == (2, 16, 16, 1)
    assert reconstruction.shape == (2, 16, 16, 3)
    assert torch.isfinite(prediction).all()
