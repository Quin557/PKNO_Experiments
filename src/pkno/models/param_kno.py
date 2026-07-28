"""Parameterized KNO variants for Stage 3.

These models are PyTorch-native.  They borrow PKNN's separation between a
shared dictionary and a parameterized Koopman family, but keep the KNO update in
Fourier space.
"""

from __future__ import annotations

import torch
from torch import nn

from pkno.dictionaries.shared_dictionary import (
    PointwiseDecoder,
    SharedPointwiseDictionary,
    StateSummaryEncoder,
)
from pkno.operators.koopman_parameterized import (
    ConditionEncoder,
    ParameterizedKoopmanOperator1D,
    ParameterizedKoopmanOperator2D,
)


class ParamKNOBase(nn.Module):
    """Common implementation for 1D and 2D Parameterized-KNO."""

    spatial_dim: int

    def __init__(
        self,
        *,
        spatial_dim: int,
        input_channels: int,
        output_channels: int,
        condition_dim: int,
        observable_dim: int = 32,
        modes: int = 16,
        decompose: int = 8,
        dictionary_hidden_dim: int = 128,
        dictionary_depth: int = 2,
        decoder_hidden_dim: int = 128,
        condition_embed_dim: int = 128,
        state_embed_dim: int = 64,
        koopman_hidden_dim: int = 128,
        koopman_depth: int = 2,
        delta_scale: float = 0.05,
        linear_type: bool = True,
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

        self.dictionary = SharedPointwiseDictionary(
            input_dim=input_channels,
            observable_dim=observable_dim,
            hidden_dim=dictionary_hidden_dim,
            depth=dictionary_depth,
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
            hidden_dim=koopman_hidden_dim,
            depth=2,
        )

        if spatial_dim == 1:
            self.koopman_layer: nn.Module = ParameterizedKoopmanOperator1D(
                observable_dim=observable_dim,
                modes=modes,
                condition_embed_dim=condition_embed_dim,
                generator_hidden_dim=koopman_hidden_dim,
                generator_depth=koopman_depth,
                delta_scale=delta_scale,
            )
            self.skip = nn.Conv1d(observable_dim, observable_dim, 1)
        else:
            self.koopman_layer = ParameterizedKoopmanOperator2D(
                observable_dim=observable_dim,
                modes_x=modes,
                modes_y=modes,
                condition_embed_dim=condition_embed_dim,
                generator_hidden_dim=koopman_hidden_dim,
                generator_depth=koopman_depth,
                delta_scale=delta_scale,
            )
            self.skip = nn.Conv2d(observable_dim, observable_dim, 1)

    def _to_channels_first(self, z: torch.Tensor) -> torch.Tensor:
        if self.spatial_dim == 1:
            return z.permute(0, 2, 1)
        return z.permute(0, 3, 1, 2)

    def _to_channels_last(self, z: torch.Tensor) -> torch.Tensor:
        if self.spatial_dim == 1:
            return z.permute(0, 2, 1)
        return z.permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one step and reconstruct the input history.

        ``condition`` is the explicit finite-dimensional condition vector
        ``c_static`` or externally supplied ``c_n``.  The model also computes a
        dynamic state-condition embedding from the current history window and
        uses both pieces to generate ``K_k``.
        """

        z_shared = self.dictionary(x)
        reconstruction = self.recon_decoder(z_shared)
        state_embed = self.state_encoder(x)
        condition_embed = self.condition_encoder(condition, state_embed)

        z = self._to_channels_first(z_shared)
        z_skip = z
        koopman_weights = self.koopman_layer.make_weights(z, condition_embed)
        for _ in range(self.decompose):
            dz = self.koopman_layer(z, condition_embed, weights=koopman_weights)
            z = z + dz if self.linear_type else torch.tanh(z + dz)
        z = torch.tanh(self.skip(z_skip) + z)
        prediction = self.pred_decoder(self._to_channels_last(z))
        return prediction, reconstruction


class ParamKNO1d(ParamKNOBase):
    """Stage3_0 Parameterized-KNO for Burgers-style 1D fields."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(spatial_dim=1, **kwargs)


class ParamKNO2d(ParamKNOBase):
    """Stage3_0 Parameterized-KNO for NS and shallow-water 2D fields."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(spatial_dim=2, **kwargs)
