"""AM-PKNO models for Stage4_0.

AM-PKNO keeps Stage3's shared dictionary and parameterized Koopman family, then
replaces the truncated per-mode matrix table with AM-FNO-style neural frequency
generation.  The dictionary is intentionally condition-independent; static
physics metadata and current-state summaries condition only the generated
Koopman matrices.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from amkno.highfreq import HighFrequencyResidual1D, HighFrequencyResidual2D
from ampkno.operators import AMParamKoopmanOperator1D, AMParamKoopmanOperator2D
from pkno.dictionaries.shared_dictionary import (
    PointwiseDecoder,
    SharedPointwiseDictionary,
    StateSummaryEncoder,
)
from pkno.operators.koopman_parameterized import ConditionEncoder


class AMPKNOBase(nn.Module):
    """Common AM-PKNO implementation for 1D and 2D scalar PDE rollout."""

    def __init__(
        self,
        *,
        spatial_dim: int,
        input_channels: int,
        output_channels: int,
        condition_dim: int,
        observable_dim: int = 32,
        decompose: int = 8,
        max_modes: int = 0,
        frequency_basis_dim: int = 32,
        dictionary_hidden_dim: int = 128,
        dictionary_depth: int = 2,
        basis_kind: str = "generic",
        decoder_hidden_dim: int = 128,
        condition_embed_dim: int = 128,
        state_embed_dim: int = 64,
        generator_hidden_dim: int = 128,
        generator_depth: int = 2,
        output_scale: float = 0.02,
        operator_factorization: str = "factorized",
        factorized_rank: int = 1,
        factorized_input_chunk: int = 4,
        linear_type: bool = True,
        use_hf_residual: bool = False,
        hf_hidden_dim: int = 32,
        hf_residual_scale: float = 0.1,
        checkpoint_koopman: bool = True,
    ) -> None:
        super().__init__()
        if spatial_dim not in {1, 2}:
            raise ValueError("spatial_dim must be 1 or 2.")
        self.spatial_dim = spatial_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.condition_dim = condition_dim
        self.observable_dim = observable_dim
        self.decompose = decompose
        self.linear_type = linear_type
        self.hf_residual_scale = hf_residual_scale
        self.checkpoint_koopman = checkpoint_koopman

        self.dictionary = SharedPointwiseDictionary(
            input_dim=input_channels,
            observable_dim=observable_dim,
            hidden_dim=dictionary_hidden_dim,
            depth=dictionary_depth,
            basis_kind=basis_kind,
        )
        self.pred_decoder = PointwiseDecoder(
            observable_dim=observable_dim,
            output_dim=output_channels,
            hidden_dim=decoder_hidden_dim,
            depth=1,
        )
        self.recon_decoder = PointwiseDecoder(
            observable_dim=observable_dim,
            output_dim=input_channels,
            hidden_dim=decoder_hidden_dim,
            depth=1,
        )
        self.state_encoder = StateSummaryEncoder(
            input_dim=input_channels,
            embed_dim=state_embed_dim,
            hidden_dim=dictionary_hidden_dim,
        )
        self.condition_encoder = ConditionEncoder(
            condition_dim=condition_dim,
            state_embed_dim=state_embed_dim,
            output_dim=condition_embed_dim,
            hidden_dim=generator_hidden_dim,
            depth=2,
        )

        if spatial_dim == 1:
            self.koopman_layer: nn.Module = AMParamKoopmanOperator1D(
                observable_dim=observable_dim,
                max_modes=max_modes,
                frequency_basis_dim=frequency_basis_dim,
                condition_embed_dim=condition_embed_dim,
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
            self.koopman_layer = AMParamKoopmanOperator2D(
                observable_dim=observable_dim,
                max_modes=max_modes,
                frequency_basis_dim=frequency_basis_dim,
                condition_embed_dim=condition_embed_dim,
                generator_hidden_dim=generator_hidden_dim,
                generator_depth=generator_depth,
                output_scale=output_scale,
                operator_factorization=operator_factorization,
                factorized_rank=factorized_rank,
                factorized_input_chunk=factorized_input_chunk,
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

    def _koopman_update(
        self,
        z: torch.Tensor,
        condition_embed: torch.Tensor,
        weights: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if not self.checkpoint_koopman or not self.training:
            return self.koopman_layer(z, condition_embed, weights=weights)

        if isinstance(weights, tuple):
            factor_x, factor_y = weights

            def run_factorized(
                z_in: torch.Tensor,
                factor_x_in: torch.Tensor,
                factor_y_in: torch.Tensor,
            ) -> torch.Tensor:
                return self.koopman_layer(z_in, condition_embed, weights=(factor_x_in, factor_y_in))

            return checkpoint(run_factorized, z, factor_x, factor_y, use_reentrant=False)

        def run_full(z_in: torch.Tensor, weights_in: torch.Tensor) -> torch.Tensor:
            return self.koopman_layer(z_in, condition_embed, weights=weights_in)

        return checkpoint(run_full, z, weights, use_reentrant=False)

    def forward(self, history: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one autoregressive step and reconstruct the current history."""

        z_shared = self.dictionary(history)
        reconstruction = self.recon_decoder(z_shared)
        state_embed = self.state_encoder(history)
        condition_embed = self.condition_encoder(condition, state_embed)

        z = self._to_channels_first(z_shared)
        z_skip = z
        weights = self.koopman_layer.make_weights(z, condition_embed)
        for _ in range(self.decompose):
            dz = self._koopman_update(z, condition_embed, weights)
            z = z + dz if self.linear_type else torch.tanh(z + dz)
        z = torch.tanh(self.skip(z_skip) + z)
        prediction = self.pred_decoder(self._to_channels_last(z))
        if self.hf_residual is not None:
            prediction = prediction + self.hf_residual_scale * self.hf_residual(history)
        return prediction, reconstruction


class AMPKNO1d(AMPKNOBase):
    """Stage4_0 AM-PKNO for Burgers-style 1D fields."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(spatial_dim=1, **kwargs)


class AMPKNO2d(AMPKNOBase):
    """Stage4_0 AM-PKNO for NS and shallow-water 2D fields."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(spatial_dim=2, **kwargs)
