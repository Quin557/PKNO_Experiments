"""AM-style frequency-generated Koopman operators."""

from __future__ import annotations

import torch
from torch import nn

from amkno.frequency import (
    AmortizedMatrixGenerator,
    ChebyshevFrequencyEmbedding,
    FactorizedAmortizedMatrixGenerator2D,
)


def _cap_count(size: int, max_modes: int | None) -> int:
    if max_modes is None or max_modes <= 0:
        return size
    return min(size, max_modes)


class AMKoopmanOperator1D(nn.Module):
    """1D Fourier Koopman update with generated matrices ``K_k = G(e(k))``.

    Unlike fixed KNO, retained modes do not each own an independent matrix.
    ``max_modes`` is therefore a compute cap.  ``max_modes <= 0`` uses every
    frequency available from ``rfft`` at the current grid resolution.
    """

    def __init__(
        self,
        observable_dim: int,
        *,
        max_modes: int = 0,
        frequency_basis_dim: int = 32,
        condition_embed_dim: int = 0,
        generator_hidden_dim: int = 128,
        generator_depth: int = 2,
        output_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.observable_dim = observable_dim
        self.max_modes = max_modes
        self.embedding = ChebyshevFrequencyEmbedding(spatial_dim=1, basis_dim=frequency_basis_dim)
        self.generator = AmortizedMatrixGenerator(
            freq_embed_dim=self.embedding.output_dim,
            observable_dim=observable_dim,
            condition_dim=condition_embed_dim,
            hidden_dim=generator_hidden_dim,
            depth=generator_depth,
            output_scale=output_scale,
        )

    @staticmethod
    def _time_march(x_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # x_ft: [B, O_in, M], weights: [M, O_in, O_out] or [B, M, O_in, O_out].
        if weights.ndim == 3:
            return torch.einsum("bim,mio->bom", x_ft, weights)
        return torch.einsum("bim,bmio->bom", x_ft, weights)

    def _freq(self, spatial_size: int, modes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.fft.rfftfreq(spatial_size, device=device, dtype=dtype)[:modes].unsqueeze(-1)

    def make_weights(self, x: torch.Tensor, condition_embed: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"1D operator expects [B, O, X], got {tuple(x.shape)}.")
        modes = _cap_count(torch.fft.rfft(x).shape[-1], self.max_modes)
        freq_embed = self.embedding(self._freq(x.shape[-1], modes, x.device, x.real.dtype))
        return self.generator(freq_embed, condition_embed)

    def forward(
        self,
        x: torch.Tensor,
        condition_embed: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"1D operator expects [B, O, X], got {tuple(x.shape)}.")
        x_ft = torch.fft.rfft(x)
        if weights is None:
            weights = self.make_weights(x, condition_embed)
        modes = weights.shape[-3] if weights.ndim == 3 else weights.shape[1]
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :modes] = self._time_march(x_ft[:, :, :modes], weights)
        return torch.fft.irfft(out_ft, n=x.shape[-1])


class AMKoopmanOperator2D(nn.Module):
    """2D Fourier Koopman update with AM-FNO-style generated matrices.

    The default ``operator_factorization='factorized'`` uses separated x/y
    frequency generators and combines their complex factors during time
    marching.  This is closer to AM-FNO's MLP implementation and avoids running
    a large generator over every Cartesian-product frequency pair.
    """

    def __init__(
        self,
        observable_dim: int,
        *,
        max_modes: int = 0,
        frequency_basis_dim: int = 32,
        condition_embed_dim: int = 0,
        generator_hidden_dim: int = 128,
        generator_depth: int = 2,
        output_scale: float = 0.05,
        operator_factorization: str = "factorized",
        factorized_rank: int = 1,
    ) -> None:
        super().__init__()
        if operator_factorization not in {"factorized", "full"}:
            raise ValueError("operator_factorization must be 'factorized' or 'full'.")
        if operator_factorization == "factorized" and condition_embed_dim > 0:
            raise ValueError(
                "The factorized 2D generator is frequency-only in Stage1_0. "
                "Use operator_factorization='full' for state-conditioned ablations."
            )
        self.observable_dim = observable_dim
        self.max_modes = max_modes
        self.operator_factorization = operator_factorization
        if operator_factorization == "factorized":
            self.axis_embedding = ChebyshevFrequencyEmbedding(spatial_dim=1, basis_dim=frequency_basis_dim)
            self.factorized_generator = FactorizedAmortizedMatrixGenerator2D(
                freq_embed_dim=self.axis_embedding.output_dim,
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
            self.generator = AmortizedMatrixGenerator(
                freq_embed_dim=self.embedding.output_dim,
                observable_dim=observable_dim,
                condition_dim=condition_embed_dim,
                hidden_dim=generator_hidden_dim,
                depth=generator_depth,
                output_scale=output_scale,
            )
            self.axis_embedding = None
            self.factorized_generator = None

    @staticmethod
    def _time_march(x_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # x_ft: [B, O_in, MX, MY], weights: [MX, MY, O_in, O_out] or [B, MX, MY, O_in, O_out].
        if weights.ndim == 4:
            return torch.einsum("bixy,xyio->boxy", x_ft, weights)
        return torch.einsum("bixy,bxyio->boxy", x_ft, weights)

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

    def make_weights(
        self,
        x: torch.Tensor,
        condition_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"2D operator expects [B, O, X, Y], got {tuple(x.shape)}.")
        x_ft = torch.fft.rfft2(x)
        idx_x, idx_y = self._indices(x_ft.shape[-2], x_ft.shape[-1], x.device)
        if self.operator_factorization == "factorized":
            if condition_embed is not None:
                raise ValueError("Factorized Stage1_0 AM-KNO is frequency-only.")
            if self.axis_embedding is None or self.factorized_generator is None:
                raise RuntimeError("Factorized generator is not initialized.")
            freq_x, freq_y = self._axis_freq(x.shape[-2], x.shape[-1], idx_x, idx_y, x.device, x.real.dtype)
            factor_x, factor_y = self.factorized_generator(self.axis_embedding(freq_x), self.axis_embedding(freq_y))
            # Materialize the frequency-only K once per rollout step, then reuse
            # it across all Koopman decompose iterations.
            return torch.einsum("xior,yior->xyio", factor_x, factor_y)

        if self.embedding is None or self.generator is None:
            raise RuntimeError("Full generator is not initialized.")
        freq = self._freq(x.shape[-2], x.shape[-1], idx_x, idx_y, x.device, x.real.dtype)
        return self.generator(self.embedding(freq), condition_embed)

    def forward(
        self,
        x: torch.Tensor,
        condition_embed: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"2D operator expects [B, O, X, Y], got {tuple(x.shape)}.")
        x_ft = torch.fft.rfft2(x)
        idx_x, idx_y = self._indices(x_ft.shape[-2], x_ft.shape[-1], x.device)
        if weights is None:
            weights = self.make_weights(x, condition_embed)
        selected = x_ft[:, :, idx_x][:, :, :, idx_y]
        out_ft = torch.zeros_like(x_ft)
        marched = self._time_march(selected, weights)
        out_ft[:, :, idx_x[:, None], idx_y[None, :]] = marched
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])
