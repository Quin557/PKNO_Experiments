"""AM-PKNO Fourier Koopman operators."""

from __future__ import annotations

import torch
from torch import nn

from ampkno.frequency import (
    ChebyshevFrequencyEmbedding,
    ConditionedFactorizedMatrixGenerator2D,
    ConditionedMatrixGenerator,
)


FactorizedWeights2D = tuple[torch.Tensor, torch.Tensor]


def _cap_count(size: int, max_modes: int | None) -> int:
    if max_modes is None or max_modes <= 0:
        return size
    return min(size, max_modes)


class AMParamKoopmanOperator1D(nn.Module):
    """1D ``K(k, c_n)`` generated from Chebyshev frequency bases and conditions."""

    def __init__(
        self,
        observable_dim: int,
        *,
        max_modes: int = 0,
        frequency_basis_dim: int = 32,
        condition_embed_dim: int = 128,
        generator_hidden_dim: int = 128,
        generator_depth: int = 2,
        output_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.observable_dim = observable_dim
        self.max_modes = max_modes
        self.embedding = ChebyshevFrequencyEmbedding(spatial_dim=1, basis_dim=frequency_basis_dim)
        self.generator = ConditionedMatrixGenerator(
            freq_embed_dim=self.embedding.output_dim,
            condition_embed_dim=condition_embed_dim,
            observable_dim=observable_dim,
            hidden_dim=generator_hidden_dim,
            depth=generator_depth,
            output_scale=output_scale,
        )

    @staticmethod
    def _time_march(x_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bim,bmio->bom", x_ft, weights)

    def _freq(self, spatial_size: int, modes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.fft.rfftfreq(spatial_size, device=device, dtype=dtype)[:modes].unsqueeze(-1)

    def make_weights(self, x: torch.Tensor, condition_embed: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"1D operator expects [B, O, X], got {tuple(x.shape)}.")
        modes = _cap_count(x.shape[-1] // 2 + 1, self.max_modes)
        freq_embed = self.embedding(self._freq(x.shape[-1], modes, x.device, x.real.dtype))
        return self.generator(freq_embed, condition_embed)

    def forward(
        self,
        x: torch.Tensor,
        condition_embed: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"1D operator expects [B, O, X], got {tuple(x.shape)}.")
        x_ft = torch.fft.rfft(x)
        if weights is None:
            weights = self.make_weights(x, condition_embed)
        modes = weights.shape[1]
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :modes] = self._time_march(x_ft[:, :, :modes], weights)
        return torch.fft.irfft(out_ft, n=x.shape[-1])


class AMParamKoopmanOperator2D(nn.Module):
    """2D conditioned AM Koopman operator.

    The default factorized path avoids running a large MLP over every ``(kx, ky)``
    pair.  Because Stage4_0 is parameterized, the x/y factors still depend on the
    batch condition embedding ``c_n``.
    """

    def __init__(
        self,
        observable_dim: int,
        *,
        max_modes: int = 0,
        frequency_basis_dim: int = 32,
        condition_embed_dim: int = 128,
        generator_hidden_dim: int = 128,
        generator_depth: int = 2,
        output_scale: float = 0.02,
        operator_factorization: str = "factorized",
        factorized_rank: int = 1,
    ) -> None:
        super().__init__()
        if operator_factorization not in {"factorized", "full"}:
            raise ValueError("operator_factorization must be 'factorized' or 'full'.")
        self.observable_dim = observable_dim
        self.max_modes = max_modes
        self.operator_factorization = operator_factorization
        if operator_factorization == "factorized":
            self.axis_embedding = ChebyshevFrequencyEmbedding(spatial_dim=1, basis_dim=frequency_basis_dim)
            self.factorized_generator = ConditionedFactorizedMatrixGenerator2D(
                freq_embed_dim=self.axis_embedding.output_dim,
                condition_embed_dim=condition_embed_dim,
                observable_dim=observable_dim,
                rank=factorized_rank,
                hidden_dim=generator_hidden_dim,
                depth=generator_depth,
                output_scale=output_scale,
            )
            self.embedding = None
            self.generator = None
        else:
            self.embedding = ChebyshevFrequencyEmbedding(spatial_dim=2, basis_dim=frequency_basis_dim)
            self.generator = ConditionedMatrixGenerator(
                freq_embed_dim=self.embedding.output_dim,
                condition_embed_dim=condition_embed_dim,
                observable_dim=observable_dim,
                hidden_dim=generator_hidden_dim,
                depth=generator_depth,
                output_scale=output_scale,
            )
            self.axis_embedding = None
            self.factorized_generator = None

    @staticmethod
    def _time_march_full(x_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,bxyio->boxy", x_ft, weights)

    @staticmethod
    def _time_march_factorized(x_ft: torch.Tensor, weights: FactorizedWeights2D) -> torch.Tensor:
        factor_x, factor_y = weights
        if x_ft.shape[0] != factor_x.shape[0] or x_ft.shape[0] != factor_y.shape[0]:
            raise ValueError("x_ft and factorized weights must share the same batch size.")
        if x_ft.shape[1] != factor_x.shape[2] or x_ft.shape[1] != factor_y.shape[2]:
            raise ValueError("x_ft and factorized weights disagree on observable input dimension.")
        if factor_x.shape[-1] != factor_y.shape[-1]:
            raise ValueError("x/y factorized weights disagree on rank.")

        # STAGE4_OOM_FIX_MEMORY_EFFICIENT_CONTRACTION:
        # A direct three-operand einsum,
        #   "bixy,bxior,byior->boxy",
        # lets PyTorch materialize a [B, X, Y, I, O, R]-scale intermediate on
        # some contraction paths. In a 40-step rollout with decompose=8 this
        # intermediate is saved hundreds of times for autograd and exhausts
        # 48GB GPUs. Accumulating one input observable/rank at a time keeps the
        # largest temporary at output size [B, O, X, Y] while preserving the same
        # mathematical factorization.
        batch, _, modes_x, modes_y = x_ft.shape
        out_channels = factor_x.shape[3]
        rank = factor_x.shape[-1]
        out = x_ft.new_zeros((batch, out_channels, modes_x, modes_y))
        for rank_index in range(rank):
            for input_index in range(x_ft.shape[1]):
                x_i = x_ft[:, input_index].unsqueeze(1)
                fx_i = factor_x[:, :, input_index, :, rank_index].permute(0, 2, 1).unsqueeze(-1)
                fy_i = factor_y[:, :, input_index, :, rank_index].permute(0, 2, 1).unsqueeze(-2)
                out = out + x_i * fx_i * fy_i
        return out

    def _indices(self, size_x: int, size_yh: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_modes is None or self.max_modes <= 0:
            return torch.arange(size_x, device=device), torch.arange(size_yh, device=device)
        pos = min(size_x, self.max_modes)
        neg = min(max(size_x - pos, 0), self.max_modes)
        if neg > 0:
            idx_x = torch.cat(
                [torch.arange(pos, device=device), torch.arange(size_x - neg, size_x, device=device)]
            )
        else:
            idx_x = torch.arange(pos, device=device)
        idx_y = torch.arange(min(size_yh, self.max_modes), device=device)
        return idx_x, idx_y

    def _axis_freq(
        self,
        size_x: int,
        size_y: int,
        idx_x: torch.Tensor,
        idx_y: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fx = torch.fft.fftfreq(size_x, device=device, dtype=dtype)[idx_x].unsqueeze(-1)
        fy = torch.fft.rfftfreq(size_y, device=device, dtype=dtype)[idx_y].unsqueeze(-1)
        return fx, fy

    def _freq(
        self,
        size_x: int,
        size_y: int,
        idx_x: torch.Tensor,
        idx_y: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        fx = torch.fft.fftfreq(size_x, device=device, dtype=dtype)[idx_x]
        fy = torch.fft.rfftfreq(size_y, device=device, dtype=dtype)[idx_y]
        grid_x, grid_y = torch.meshgrid(fx, fy, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1)

    def make_weights(self, x: torch.Tensor, condition_embed: torch.Tensor) -> torch.Tensor | FactorizedWeights2D:
        if x.ndim != 4:
            raise ValueError(f"2D operator expects [B, O, X, Y], got {tuple(x.shape)}.")
        idx_x, idx_y = self._indices(x.shape[-2], x.shape[-1] // 2 + 1, x.device)
        if self.operator_factorization == "factorized":
            if self.axis_embedding is None or self.factorized_generator is None:
                raise RuntimeError("Factorized generator is not initialized.")
            freq_x, freq_y = self._axis_freq(x.shape[-2], x.shape[-1], idx_x, idx_y, x.device, x.real.dtype)
            return self.factorized_generator(
                self.axis_embedding(freq_x),
                self.axis_embedding(freq_y),
                condition_embed,
            )

        if self.embedding is None or self.generator is None:
            raise RuntimeError("Full generator is not initialized.")
        freq = self._freq(x.shape[-2], x.shape[-1], idx_x, idx_y, x.device, x.real.dtype)
        return self.generator(self.embedding(freq), condition_embed)

    def forward(
        self,
        x: torch.Tensor,
        condition_embed: torch.Tensor,
        weights: torch.Tensor | FactorizedWeights2D | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"2D operator expects [B, O, X, Y], got {tuple(x.shape)}.")
        x_ft = torch.fft.rfft2(x)
        idx_x, idx_y = self._indices(x_ft.shape[-2], x_ft.shape[-1], x.device)
        if weights is None:
            weights = self.make_weights(x, condition_embed)
        use_all_modes = idx_x.numel() == x_ft.shape[-2] and idx_y.numel() == x_ft.shape[-1]
        selected = x_ft if use_all_modes else x_ft[:, :, idx_x][:, :, :, idx_y]
        marched = (
            self._time_march_factorized(selected, weights)
            if isinstance(weights, tuple)
            else self._time_march_full(selected, weights)
        )
        if use_all_modes:
            out_ft = marched
        else:
            out_ft = torch.zeros_like(x_ft)
            out_ft[:, :, idx_x[:, None], idx_y[None, :]] = marched
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])
