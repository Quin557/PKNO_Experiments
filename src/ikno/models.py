"""Standalone Invertible Koopman Neural Operator baseline models."""

from __future__ import annotations

import torch
from torch import nn

from ikno.invertible import InvertibleDictionary
from ikno.operators import FixedKoopmanOperator1D, FixedKoopmanOperator2D


class IKNOBase(nn.Module):
    """IKNO for scalar autoregressive PDE rollout on one- or two-dimensional grids."""

    def __init__(
        self,
        *,
        spatial_dim: int,
        input_channels: int,
        output_channels: int,
        observable_dim: int = 32,
        modes: int = 16,
        operator_layers: int = 4,
        koopman_power: int = 2,
        inn_blocks: int = 4,
        inn_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if spatial_dim not in {1, 2}:
            raise ValueError("spatial_dim must be 1 or 2.")
        if output_channels > input_channels:
            raise ValueError("IKNO expects output channels to be drawn from its input history.")
        if operator_layers <= 0 or koopman_power <= 0:
            raise ValueError("operator_layers and koopman_power must be positive.")
        self.spatial_dim = spatial_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.observable_dim = observable_dim
        self.operator_layers = operator_layers
        self.koopman_power = koopman_power
        self.dictionary = InvertibleDictionary(
            input_channels,
            observable_dim,
            blocks=inn_blocks,
            hidden_dim=inn_hidden_dim,
        )
        if spatial_dim == 1:
            self.koopman: nn.Module = FixedKoopmanOperator1D(observable_dim, modes)
            self.high_frequency = nn.Conv1d(observable_dim, observable_dim, kernel_size=1)
        else:
            self.koopman = FixedKoopmanOperator2D(observable_dim, modes, modes)
            self.high_frequency = nn.Conv2d(observable_dim, observable_dim, kernel_size=1)
        self.activation = nn.GELU()

    def _to_channels_first(self, z: torch.Tensor) -> torch.Tensor:
        if self.spatial_dim == 1:
            return z.permute(0, 2, 1)
        return z.permute(0, 3, 1, 2)

    def _to_channels_last(self, z: torch.Tensor) -> torch.Tensor:
        if self.spatial_dim == 1:
            return z.permute(0, 2, 1)
        return z.permute(0, 2, 3, 1)

    def forward(
        self,
        history: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one autoregressive step with the paper's IKNO update.

        ``condition`` is accepted only to share the repository's loader and
        rollout interfaces.  IKNO is a non-parameterized baseline and does not
        condition its Koopman matrix on this vector.
        """

        del condition
        z = self.dictionary(history)
        z_cf = self._to_channels_first(z)
        for _ in range(self.operator_layers):
            low_frequency = self.koopman(z_cf, power=self.koopman_power)
            z_cf = self.activation(low_frequency + self.high_frequency(z_cf))
        decoded_history = self.dictionary.inverse(self._to_channels_last(z_cf))
        prediction = decoded_history[..., -self.output_channels :]
        # The trainer's two-value interface is shared with KNO-family models.
        # Returning history makes its diagnostic reconstruction exact; IKNO
        # training config sets reconstruction weight to zero.
        return prediction, history


class IKNO1d(IKNOBase):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(spatial_dim=1, **kwargs)


class IKNO2d(IKNOBase):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(spatial_dim=2, **kwargs)
