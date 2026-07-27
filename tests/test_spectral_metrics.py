import torch

from pkno.metrics.spectral import gradient_relative_l2, spectral_band_relative_l2


def test_spectral_metrics_are_zero_for_identical_inputs():
    x = torch.randn(2, 16, 16, 1)
    metrics = spectral_band_relative_l2(x, x)
    assert metrics["low_band_spectral_rel_l2"].item() == 0.0
    assert metrics["mid_band_spectral_rel_l2"].item() == 0.0
    assert metrics["high_band_spectral_rel_l2"].item() == 0.0
    assert metrics["high_frequency_energy_ratio_error"].item() == 0.0
    assert gradient_relative_l2(x, x).item() == 0.0


def test_spectral_metrics_support_1d_channels_last_inputs():
    x = torch.randn(2, 32, 3)
    y = x + 0.01 * torch.randn_like(x)
    metrics = spectral_band_relative_l2(y, x)
    assert set(metrics) == {
        "low_band_spectral_rel_l2",
        "mid_band_spectral_rel_l2",
        "high_band_spectral_rel_l2",
        "high_frequency_energy_ratio_error",
    }
    assert gradient_relative_l2(y, x).isfinite()
