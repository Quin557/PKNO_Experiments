import pytest

torch = pytest.importorskip("torch")

from pkno_v2.model import PKNOV2_1d, PKNOV2_2d


def test_pkno_v2_1d_shape():
    model = PKNOV2_1d(input_channels=1, output_channels=1, condition_dim=4, observable_dim=32, modes=4, decompose=2)
    out = model(torch.randn(2, 16, 1), torch.randn(2, 4))
    assert out.prediction.shape == (2, 16, 1)
    assert out.reconstruction.shape == (2, 16, 1)


def test_pkno_v2_2d_shape():
    model = PKNOV2_2d(input_channels=10, output_channels=1, condition_dim=5, observable_dim=32, modes=4, decompose=2)
    out = model(torch.randn(2, 8, 8, 10), torch.randn(2, 5))
    assert out.prediction.shape == (2, 8, 8, 1)
    assert out.reconstruction.shape == (2, 8, 8, 10)
