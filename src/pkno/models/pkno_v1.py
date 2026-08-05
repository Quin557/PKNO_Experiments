"""Stage3_2 PKNO_v1: residual fields with physics-first conditioning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pkno.dictionaries.shared_dictionary import PointwiseDecoder, SharedPointwiseDictionary, StateSummaryEncoder
from pkno.operators.koopman_parameterized import ParameterizedKoopmanOperator1D, ParameterizedKoopmanOperator2D


@dataclass
class PKNOV1Output:
    prediction: torch.Tensor
    reconstruction: torch.Tensor
    state_correction: torch.Tensor
    gate: torch.Tensor


class PhysicsFirstConditionEncoder(nn.Module):
    """Let known physical conditions anchor the generated operator.

    The state summary is intentionally bounded before it can perturb the
    physics embedding.  This preserves the original PKNO state adaptivity
    without allowing a content-rich summary to dominate a static condition.
    """

    def __init__(self, condition_dim: int, state_dim: int, output_dim: int, hidden_dim: int, gate_max: float = 0.1) -> None:
        super().__init__()
        self.gate_max = gate_max
        self.register_buffer("condition_scale", torch.ones(condition_dim))
        self.physics = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim), nn.GELU())
        self.state = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(2 * output_dim, output_dim), nn.Sigmoid())

    def set_condition_scale(self, scale: torch.Tensor) -> None:
        if scale.shape != self.condition_scale.shape:
            raise ValueError(f"Expected condition scale shape {tuple(self.condition_scale.shape)}, got {tuple(scale.shape)}.")
        self.condition_scale.copy_(scale.detach().to(dtype=self.condition_scale.dtype).clamp_min(1e-6))

    def forward(self, condition: torch.Tensor, state_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        physics = self.physics(condition / self.condition_scale)
        state = self.state(state_embed)
        gate = self.gate_max * self.gate(torch.cat([physics, state], dim=-1))
        correction = gate * state
        return physics + correction, correction, gate


class PKNOV1Base(nn.Module):
    spatial_dim: int

    def __init__(
        self, *, spatial_dim: int, input_channels: int, output_channels: int, condition_dim: int,
        observable_dim: int = 32, modes: int = 16, decompose: int = 8, dictionary_hidden_dim: int = 128,
        dictionary_depth: int = 2, basis_kind: str = "generic", decoder_hidden_dim: int = 128,
        condition_embed_dim: int = 128, state_embed_dim: int = 64, koopman_hidden_dim: int = 128,
        koopman_depth: int = 2, delta_scale: float = 0.05, gate_max: float = 0.1,
        residual_prediction: bool = True,
    ) -> None:
        super().__init__()
        if spatial_dim not in {1, 2}:
            raise ValueError("spatial_dim must be 1 or 2.")
        self.spatial_dim = spatial_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.decompose = decompose
        self.residual_prediction = residual_prediction
        self.dictionary = SharedPointwiseDictionary(input_dim=input_channels, observable_dim=observable_dim, hidden_dim=dictionary_hidden_dim, depth=dictionary_depth, basis_kind=basis_kind)
        self.pred_decoder = PointwiseDecoder(observable_dim=observable_dim, output_dim=output_channels, hidden_dim=decoder_hidden_dim, depth=1)
        self.recon_decoder = PointwiseDecoder(observable_dim=observable_dim, output_dim=input_channels, hidden_dim=decoder_hidden_dim, depth=1)
        self.state_encoder = StateSummaryEncoder(input_dim=input_channels, embed_dim=state_embed_dim, hidden_dim=dictionary_hidden_dim)
        self.condition_encoder = PhysicsFirstConditionEncoder(condition_dim, state_embed_dim, condition_embed_dim, koopman_hidden_dim, gate_max=gate_max)
        if spatial_dim == 1:
            self.koopman_layer: nn.Module = ParameterizedKoopmanOperator1D(observable_dim, modes, condition_embed_dim, koopman_hidden_dim, koopman_depth, delta_scale)
            self.skip = nn.Conv1d(observable_dim, observable_dim, 1)
        else:
            self.koopman_layer = ParameterizedKoopmanOperator2D(observable_dim, modes, modes, condition_embed_dim, koopman_hidden_dim, koopman_depth, delta_scale)
            self.skip = nn.Conv2d(observable_dim, observable_dim, 1)

    def set_condition_scale(self, scale: torch.Tensor) -> None:
        self.condition_encoder.set_condition_scale(scale)

    def _to_channels_first(self, z: torch.Tensor) -> torch.Tensor:
        return z.permute(0, 2, 1) if self.spatial_dim == 1 else z.permute(0, 3, 1, 2)

    def _to_channels_last(self, z: torch.Tensor) -> torch.Tensor:
        return z.permute(0, 2, 1) if self.spatial_dim == 1 else z.permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> PKNOV1Output:
        z_shared = self.dictionary(x)
        reconstruction = self.recon_decoder(z_shared)
        state_embed = self.state_encoder(x)
        condition_embed, state_correction, gate = self.condition_encoder(condition, state_embed)
        z = self._to_channels_first(z_shared)
        z_skip = z
        weights = self.koopman_layer.make_weights(z, condition_embed)
        for _ in range(self.decompose):
            z = z + self.koopman_layer(z, condition_embed, weights=weights)
        z = torch.tanh(self.skip(z_skip) + z)
        delta = self.pred_decoder(self._to_channels_last(z))
        latest = x[..., -self.output_channels :]
        prediction = latest + delta if self.residual_prediction else delta
        return PKNOV1Output(prediction=prediction, reconstruction=reconstruction, state_correction=state_correction, gate=gate)


class PKNOV11d(PKNOV1Base):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(spatial_dim=1, **kwargs)


class PKNOV12d(PKNOV1Base):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(spatial_dim=2, **kwargs)
