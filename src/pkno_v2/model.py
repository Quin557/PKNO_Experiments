"""PKNO_v2-A: low-rank conditioned Fourier Koopman plus HF residual."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .conditioning import PhysicsFirstConditioner
from .dictionary import ObservableDictionary, PointwiseDecoder
from .highfreq import HighFrequencyResidual
from .operators import LowRankKoopman


@dataclass
class PKNOV2Output:
    prediction: torch.Tensor
    reconstruction: torch.Tensor
    state_gate: torch.Tensor
    eta: torch.Tensor


class _PKNOV2Base(nn.Module):
    def __init__(self, *, spatial_dim: int, input_channels: int, output_channels: int, condition_dim: int,
                 observable_dim: int = 32, modes: int = 16, decompose: int = 8, dictionary_hidden_dim: int = 128,
                 dictionary_depth: int = 2, basis_kind: str = "generic", condition_embed_dim: int = 64,
                 state_embed_dim: int = 16, koopman_hidden_dim: int = 128, koopman_depth: int = 2,
                 rank: int = 4, delta_scale: float = 0.02, eta_max: float = 0.5, hf_hidden: int = 16,
                 latent_clip: float = 5.0, residual_clip: float = 2.0) -> None:
        super().__init__()
        self.spatial_dim, self.output_channels, self.decompose = spatial_dim, output_channels, decompose
        self.latent_clip, self.residual_clip = float(latent_clip), float(residual_clip)
        self.dictionary = ObservableDictionary(input_channels, observable_dim, dictionary_hidden_dim, dictionary_depth, basis_kind)
        self.pred_decoder = PointwiseDecoder(observable_dim, output_channels, dictionary_hidden_dim)
        self.recon_decoder = PointwiseDecoder(observable_dim, input_channels, dictionary_hidden_dim)
        self.conditioner = PhysicsFirstConditioner(condition_dim, input_channels, condition_embed_dim, state_embed_dim, koopman_hidden_dim)
        self.koopman = LowRankKoopman(spatial_dim, observable_dim, modes, condition_embed_dim, rank, koopman_hidden_dim, koopman_depth, delta_scale, eta_max)
        self.highfreq = HighFrequencyResidual(spatial_dim, input_channels, output_channels, hf_hidden)

    def set_condition_scale(self, scale: torch.Tensor) -> None:
        self.conditioner.set_condition_scale(scale)

    def forward(self, history: torch.Tensor, condition: torch.Tensor) -> PKNOV2Output:
        z_shared = self.dictionary(history)
        reconstruction = self.recon_decoder(z_shared)
        cond_embed, state_gate = self.conditioner(condition, history)
        z = z_shared.permute(0, 2, 1) if self.spatial_dim == 1 else z_shared.permute(0, 3, 1, 2)
        weights = self.koopman.make_weights(z, cond_embed)
        for _ in range(self.decompose):
            z = self.koopman.step(z, cond_embed, weights, step_scale=1.0 / max(self.decompose, 1))
            z = self.latent_clip * torch.tanh(z / self.latent_clip)
        decoded = self.pred_decoder(z.permute(0, 2, 1) if self.spatial_dim == 1 else z.permute(0, 2, 3, 1))
        latest = history[..., -1:]
        # The residual branch consumes the complete time-delay history.  For
        # NS/SWE this is ten channels, while its output remains one next frame.
        delta = decoded + self.highfreq(history)
        delta = self.residual_clip * torch.tanh(delta / self.residual_clip)
        prediction = latest[..., : self.output_channels] + delta
        eta = self.koopman.eta_max * torch.sigmoid(self.koopman.raw_eta)
        return PKNOV2Output(prediction, reconstruction, state_gate, eta)


class PKNOV2_1d(_PKNOV2Base):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(spatial_dim=1, **kwargs)


class PKNOV2_2d(_PKNOV2Base):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(spatial_dim=2, **kwargs)
