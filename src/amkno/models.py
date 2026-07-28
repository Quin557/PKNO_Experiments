"""Standalone AM-KNO models for Stage1_0."""

from __future__ import annotations

import torch
from torch import nn

from amkno.dictionary import PointwiseDecoder, PointwiseEncoder, StateSummaryEncoder
from amkno.highfreq import HighFrequencyResidual1D, HighFrequencyResidual2D
from amkno.operators import AMKoopmanOperator1D, AMKoopmanOperator2D


class StaticStateConditionEncoder(nn.Module):
    """Combine optional dataset condition and dynamic state summary."""

    def __init__(
        self,
        condition_dim: int,
        state_embed_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        in_dim = condition_dim + state_embed_dim
        if in_dim <= 0:
            raise ValueError("At least one condition feature is required.")
        self.condition_dim = condition_dim
        self.state_embed_dim = state_embed_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
            activation(),
        )

    def forward(self, condition: torch.Tensor | None, state_embed: torch.Tensor | None) -> torch.Tensor:
        parts = []
        if self.condition_dim > 0:
            if condition is None or condition.shape[-1] != self.condition_dim:
                raise ValueError(f"Expected condition dim {self.condition_dim}.")
            parts.append(condition)
        if self.state_embed_dim > 0:
            if state_embed is None or state_embed.shape[-1] != self.state_embed_dim:
                raise ValueError(f"Expected state embed dim {self.state_embed_dim}.")
            parts.append(state_embed)
        return self.net(torch.cat(parts, dim=-1))


class AMKNOBase(nn.Module):
    """Common AM-KNO implementation for 1D and 2D PDE fields."""

    def __init__(
        self,
        *,
        spatial_dim: int,
        input_channels: int,
        output_channels: int,
        condition_dim: int = 0,
        observable_dim: int = 32,
        decompose: int = 8,
        max_modes: int = 0,
        frequency_basis_dim: int = 32,
        condition_mode: str = "freq",
        encoder_hidden_dim: int = 0,
        decoder_hidden_dim: int = 0,
        state_embed_dim: int = 64,
        condition_embed_dim: int = 128,
        generator_hidden_dim: int = 128,
        generator_depth: int = 2,
        output_scale: float = 0.05,
        linear_type: bool = True,
        use_hf_residual: bool = False,
        hf_hidden_dim: int = 32,
        hf_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if spatial_dim not in {1, 2}:
            raise ValueError("spatial_dim must be 1 or 2.")
        if condition_mode not in {"freq", "state", "state_static"}:
            raise ValueError("condition_mode must be 'freq', 'state', or 'state_static'.")
        self.spatial_dim = spatial_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.condition_dim = condition_dim
        self.observable_dim = observable_dim
        self.decompose = decompose
        self.condition_mode = condition_mode
        self.linear_type = linear_type
        self.hf_residual_scale = hf_residual_scale

        self.encoder = PointwiseEncoder(input_channels, observable_dim, hidden_dim=encoder_hidden_dim)
        self.pred_decoder = PointwiseDecoder(observable_dim, output_channels, hidden_dim=decoder_hidden_dim)
        self.recon_decoder = PointwiseDecoder(observable_dim, input_channels, hidden_dim=decoder_hidden_dim)

        operator_condition_dim = 0
        self.state_encoder: StateSummaryEncoder | None = None
        self.condition_encoder: StaticStateConditionEncoder | None = None
        if condition_mode != "freq":
            self.state_encoder = StateSummaryEncoder(
                input_dim=input_channels,
                embed_dim=state_embed_dim,
                hidden_dim=max(generator_hidden_dim, state_embed_dim),
            )
            static_dim = condition_dim if condition_mode == "state_static" else 0
            self.condition_encoder = StaticStateConditionEncoder(
                condition_dim=static_dim,
                state_embed_dim=state_embed_dim,
                output_dim=condition_embed_dim,
                hidden_dim=generator_hidden_dim,
            )
            operator_condition_dim = condition_embed_dim

        if spatial_dim == 1:
            self.koopman_layer: nn.Module = AMKoopmanOperator1D(
                observable_dim,
                max_modes=max_modes,
                frequency_basis_dim=frequency_basis_dim,
                condition_embed_dim=operator_condition_dim,
                generator_hidden_dim=generator_hidden_dim,
                generator_depth=generator_depth,
                output_scale=output_scale,
            )
            self.skip = nn.Conv1d(observable_dim, observable_dim, 1)
            self.hf_residual: nn.Module | None = (
                HighFrequencyResidual1D(input_channels, output_channels, hidden_dim=hf_hidden_dim)
                if use_hf_residual
                else None
            )
        else:
            self.koopman_layer = AMKoopmanOperator2D(
                observable_dim,
                max_modes=max_modes,
                frequency_basis_dim=frequency_basis_dim,
                condition_embed_dim=operator_condition_dim,
                generator_hidden_dim=generator_hidden_dim,
                generator_depth=generator_depth,
                output_scale=output_scale,
            )
            self.skip = nn.Conv2d(observable_dim, observable_dim, 1)
            self.hf_residual = (
                HighFrequencyResidual2D(input_channels, output_channels, hidden_dim=hf_hidden_dim)
                if use_hf_residual
                else None
            )

    def _to_channels_first(self, z: torch.Tensor) -> torch.Tensor:
        if self.spatial_dim == 1:
            return z.permute(0, 2, 1)
        return z.permute(0, 3, 1, 2)

    def _to_channels_last(self, z: torch.Tensor) -> torch.Tensor:
        if self.spatial_dim == 1:
            return z.permute(0, 2, 1)
        return z.permute(0, 2, 3, 1)

    def _condition_embed(self, history: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor | None:
        if self.condition_mode == "freq":
            return None
        if self.state_encoder is None or self.condition_encoder is None:
            raise RuntimeError("Condition encoders are not initialized.")
        state_embed = self.state_encoder(history)
        static_condition = condition if self.condition_mode == "state_static" else None
        return self.condition_encoder(static_condition, state_embed)

    def forward(self, history: torch.Tensor, condition: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one autoregressive step and reconstruct the current history.

        The AM idea enters only in ``self.koopman_layer``: a shared generator
        maps frequency embeddings, and optionally a state embedding, to the
        complex Koopman matrix used at that frequency.  This keeps the Stage1_0
        model separate from PKNO's parameterized/shared-dictionary code path.
        """

        z_shared = torch.tanh(self.encoder(history))
        reconstruction = self.recon_decoder(z_shared)
        condition_embed = self._condition_embed(history, condition)

        z = self._to_channels_first(z_shared)
        z_skip = z
        weights = self.koopman_layer.make_weights(z, condition_embed)
        for _ in range(self.decompose):
            dz = self.koopman_layer(z, condition_embed, weights=weights)
            z = z + dz if self.linear_type else torch.tanh(z + dz)
        z = torch.tanh(self.skip(z_skip) + z)
        prediction = self.pred_decoder(self._to_channels_last(z))
        if self.hf_residual is not None:
            prediction = prediction + self.hf_residual_scale * self.hf_residual(history)
        return prediction, reconstruction


class AMKNO1d(AMKNOBase):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(spatial_dim=1, **kwargs)


class AMKNO2d(AMKNOBase):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(spatial_dim=2, **kwargs)
