from __future__ import annotations

import torch
from torch import nn

from pkno.trainers.train_rollout import RelativeL2, rollout_model
from pkno_u.models import PKNOU2d


def test_pkno_u_autoregressive_rollout_is_finite():
    model = PKNOU2d(
        input_channels=3,
        output_channels=1,
        condition_dim=7,
        observable_dim=12,
        modes=4,
        decompose=2,
        condition_mode="physical_only",
        condition_embed_dim=16,
        koopman_hidden_dim=16,
        dictionary_hidden_dim=16,
        unet_base_channels=8,
        max_operator_norm=0.8,
    )
    output = rollout_model(
        model,
        torch.randn(2, 16, 16, 3),
        torch.randn(2, 16, 16, 3),
        torch.randn(2, 7),
        t_out=3,
        output_channels=1,
        mse=nn.MSELoss(),
        rel_l2=RelativeL2(),
    )
    assert output["pred"].shape == (2, 16, 16, 3)
    assert torch.isfinite(output["pred_mse"])
    assert torch.isfinite(output["recon_mse"])


def test_physical_only_supports_encode_once_ordered_rollout():
    model = PKNOU2d(
        input_channels=3,
        output_channels=1,
        condition_dim=7,
        observable_dim=12,
        modes=4,
        decompose=2,
        condition_mode="physical_only",
        condition_embed_dim=16,
        koopman_hidden_dim=16,
        dictionary_hidden_dim=16,
        unet_base_channels=8,
    )
    prediction = model.latent_ordered_rollout(torch.randn(2, 16, 16, 3), torch.randn(2, 7), t_out=3)
    assert prediction.shape == (2, 16, 16, 3)
    assert torch.isfinite(prediction).all()
