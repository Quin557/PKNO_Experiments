"""PKNO-U models: stable parametric Koopman propagation plus latent U-Nets."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from pkno.dictionaries.shared_dictionary import PointwiseDecoder, SharedPointwiseDictionary
from pkno_u.conditioning import CONDITION_MODES, ConditioningModule
from pkno_u.operators import StableParameterizedKoopmanOperator1D, StableParameterizedKoopmanOperator2D
from pkno_u.stability import ContractiveTransition
from pkno_u.unet import LatentUNet1D, LatentUNet2D, high_pass


class PKNOUBase(nn.Module):
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
        basis_kind: str = "generic",
        decoder_hidden_dim: int = 128,
        condition_embed_dim: int = 128,
        state_embed_dim: int = 16,
        koopman_hidden_dim: int = 128,
        koopman_depth: int = 2,
        delta_scale: float = 0.05,
        max_operator_norm: float = 0.98,
        condition_mode: str = "physical_only",
        unet_start_layer: int | None = None,
        unet_base_channels: int = 32,
        hf_cutoff: float = 0.5,
        hf_residual_scale: float = 0.05,
        checkpoint_unet: bool = True,
    ) -> None:
        super().__init__()
        if spatial_dim not in {1, 2}:
            raise ValueError("spatial_dim must be 1 or 2.")
        if decompose < 1:
            raise ValueError("decompose must be at least 1.")
        if condition_mode not in CONDITION_MODES:
            raise ValueError(f"Unknown condition_mode {condition_mode!r}.")
        self.spatial_dim = spatial_dim
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.decompose = decompose
        self.condition_mode = condition_mode
        self.hf_cutoff = hf_cutoff
        self.hf_residual_scale = hf_residual_scale
        self.checkpoint_unet = checkpoint_unet
        self.unet_start_layer = decompose // 2 if unet_start_layer is None else unet_start_layer
        if not 0 <= self.unet_start_layer <= decompose:
            raise ValueError("unet_start_layer must be in [0, decompose].")

        self.dictionary = SharedPointwiseDictionary(
            input_dim=input_channels,
            observable_dim=observable_dim,
            hidden_dim=dictionary_hidden_dim,
            depth=dictionary_depth,
            basis_kind=basis_kind,
        )
        self.pred_decoder = PointwiseDecoder(observable_dim, output_channels, hidden_dim=decoder_hidden_dim)
        self.recon_decoder = PointwiseDecoder(observable_dim, input_channels, hidden_dim=decoder_hidden_dim)
        self.conditioner = ConditioningModule(
            condition_dim=condition_dim,
            input_channels=input_channels,
            output_dim=condition_embed_dim,
            mode=condition_mode,
            hidden_dim=koopman_hidden_dim,
            state_dim=state_embed_dim,
        )
        if spatial_dim == 1:
            self.koopman_layer: nn.Module = StableParameterizedKoopmanOperator1D(
                observable_dim=observable_dim,
                modes=modes,
                condition_embed_dim=condition_embed_dim,
                hidden_dim=koopman_hidden_dim,
                depth=koopman_depth,
                delta_scale=delta_scale,
                max_operator_norm=max_operator_norm,
            )
            unet_cls: type[nn.Module] = LatentUNet1D
            self.skip = nn.Conv1d(observable_dim, observable_dim, 1)
        else:
            self.koopman_layer = StableParameterizedKoopmanOperator2D(
                observable_dim=observable_dim,
                modes_x=modes,
                modes_y=modes,
                condition_embed_dim=condition_embed_dim,
                hidden_dim=koopman_hidden_dim,
                depth=koopman_depth,
                delta_scale=delta_scale,
                max_operator_norm=max_operator_norm,
            )
            unet_cls = LatentUNet2D
            self.skip = nn.Conv2d(observable_dim, observable_dim, 1)
        self.unet_layers = nn.ModuleList(
            [unet_cls(observable_dim, base_channels=unet_base_channels) for _ in range(decompose - self.unet_start_layer)]
        )
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def _to_channels_first(self, z: torch.Tensor) -> torch.Tensor:
        return z.permute(0, 2, 1) if self.spatial_dim == 1 else z.permute(0, 3, 1, 2)

    def _to_channels_last(self, z: torch.Tensor) -> torch.Tensor:
        return z.permute(0, 2, 1) if self.spatial_dim == 1 else z.permute(0, 2, 3, 1)

    def _unet_residual(self, unet: nn.Module, z: torch.Tensor) -> torch.Tensor:
        if self.training and self.checkpoint_unet:
            raw = checkpoint(unet, z, use_reentrant=False)
        else:
            raw = unet(z)
        return high_pass(raw, cutoff=self.hf_cutoff)

    def _advance_latent(self, z: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one physical step in the latent field without re-encoding."""

        z_skip = z
        unet_norm = z.new_tensor(0.0)
        for layer_index in range(self.decompose):
            z_input = z
            z = self.koopman_layer(z_input, weights)
            if layer_index >= self.unet_start_layer:
                unet = self.unet_layers[layer_index - self.unet_start_layer]
                hf_update = self._unet_residual(unet, z_input)
                z = z + self.hf_residual_scale * hf_update
                unet_norm = unet_norm + hf_update.square().mean().sqrt()
        return torch.tanh(self.skip(z_skip) + z), unet_norm

    def _record_diagnostics(
        self,
        *,
        weights: torch.Tensor,
        z: torch.Tensor,
        gate: torch.Tensor,
        unet_norm: torch.Tensor,
    ) -> None:
        matrix_norms = torch.linalg.matrix_norm(weights, ord=2)
        self.last_diagnostics = {
            "condition_gate": gate.detach().mean(),
            "matrix_spectral_mean": matrix_norms.detach().mean(),
            "matrix_spectral_max": matrix_norms.detach().max(),
            "latent_rms": z.detach().square().mean().sqrt(),
            "unet_highpass_rms": (unet_norm / max(len(self.unet_layers), 1)).detach(),
        }

    def forward(self, history: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_shared = self.dictionary(history)
        reconstruction = self.recon_decoder(z_shared)
        condition_embed, gate = self.conditioner(condition, history)
        z = self._to_channels_first(z_shared)
        weights = self.koopman_layer.make_weights(z, condition_embed)
        z, unet_norm = self._advance_latent(z, weights)
        prediction = self.pred_decoder(self._to_channels_last(z))
        self._record_diagnostics(weights=weights, z=z, gate=gate, unet_norm=unet_norm)
        return prediction, reconstruction

    def latent_ordered_rollout(
        self,
        history: torch.Tensor,
        condition: torch.Tensor,
        *,
        t_out: int,
    ) -> torch.Tensor:
        """Encode once then apply a fixed physical-condition propagator in order.

        This deliberately only supports model A.  State-conditioned variants
        require a new history-derived condition after each step and therefore do
        not represent a fixed matrix product in shared coordinates.
        """

        if self.condition_mode != "physical_only":
            raise ValueError("latent_ordered_rollout is valid only for condition_mode='physical_only'.")
        if t_out < 1:
            raise ValueError("t_out must be positive.")
        z_shared = self.dictionary(history)
        condition_embed, gate = self.conditioner(condition, history)
        z = self._to_channels_first(z_shared)
        weights = self.koopman_layer.make_weights(z, condition_embed)
        predictions: list[torch.Tensor] = []
        for _ in range(t_out):
            z, unet_norm = self._advance_latent(z, weights)
            predictions.append(self.pred_decoder(self._to_channels_last(z)))
        self._record_diagnostics(weights=weights, z=z, gate=gate, unet_norm=unet_norm)
        if self.output_channels == 1:
            return torch.cat(predictions, dim=-1)
        return torch.stack(predictions, dim=-2)


class PKNOU1d(PKNOUBase):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(spatial_dim=1, **kwargs)


class PKNOU2d(PKNOUBase):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(spatial_dim=2, **kwargs)
