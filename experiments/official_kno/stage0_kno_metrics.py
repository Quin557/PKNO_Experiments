from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


EPS = 1e-12
DEFAULT_BANDS = {
    "low": [0.0, 0.33],
    "mid": [0.33, 0.66],
    "high": [0.66, 1.01],
}


def to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_builtin(obj), indent=2, sort_keys=True), encoding="utf-8")


def count_params(model: nn.Module) -> int:
    return sum(param.numel() * (2 if param.is_complex() else 1) for param in model.parameters())


class RelativeL2:
    def __init__(self, eps: float = EPS) -> None:
        self.eps = eps

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch = pred.shape[0]
        diff = torch.linalg.vector_norm((pred - target).reshape(batch, -1), dim=1)
        denom = torch.linalg.vector_norm(target.reshape(batch, -1), dim=1).clamp_min(self.eps)
        return diff / denom


def _target_step(target: torch.Tensor, step: int, output_channels: int) -> torch.Tensor:
    if output_channels == 1:
        if target.shape[-1] <= step:
            raise ValueError(f"Target has {target.shape[-1]} rollout channels, cannot read step {step}.")
        return target[..., step : step + 1]
    if target.ndim < 3:
        raise ValueError(f"Unexpected target shape {tuple(target.shape)}.")
    if target.shape[-1] != output_channels:
        raise ValueError(
            f"Expected target output channels {output_channels}, got last dim {target.shape[-1]}."
        )
    if target.shape[-2] <= step:
        raise ValueError(f"Target has {target.shape[-2]} time steps, cannot read step {step}.")
    return target[..., step, :]


def _append_prediction(history: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if history.shape[-1] == pred.shape[-1]:
        return pred
    if pred.shape[-1] != 1:
        raise ValueError(
            "Autoregressive history update only supports scalar outputs when "
            "history has more channels than the prediction."
        )
    return torch.cat([history[..., 1:], pred], dim=-1)


def rollout_model(
    model: nn.Module,
    initial_history: torch.Tensor,
    target: torch.Tensor,
    *,
    t_out: int,
    output_channels: int,
    mse: nn.Module,
    rel_l2: RelativeL2,
) -> dict[str, torch.Tensor]:
    history = initial_history
    pred_steps: list[torch.Tensor] = []
    pred_mse = history.new_tensor(0.0)
    recon_mse = history.new_tensor(0.0)
    step_rel = history.new_tensor(0.0)

    for step in range(t_out):
        pred_step, recon = model(history)
        truth_step = _target_step(target, step, output_channels)
        pred_mse = pred_mse + mse(pred_step, truth_step)
        recon_mse = recon_mse + mse(recon, history)
        step_rel = step_rel + rel_l2(pred_step, truth_step).sum()
        pred_steps.append(pred_step if output_channels == 1 else pred_step.unsqueeze(-2))
        history = _append_prediction(history, pred_step)

    pred = torch.cat(pred_steps, dim=-1 if output_channels == 1 else -2)
    return {
        "pred": pred,
        "pred_mse": pred_mse,
        "recon_mse": recon_mse,
        "step_rel": step_rel,
    }


def predict_burgers(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    pred, _ = model(x)
    return pred.unsqueeze(-1) if pred.ndim == x.ndim - 1 else pred


def predict_rollout(model: nn.Module, x: torch.Tensor, t_out: int) -> torch.Tensor:
    preds = []
    history = x
    for _ in range(t_out):
        im, _ = model(history)
        step = im[..., -1:]
        preds.append(step)
        history = torch.cat((history[..., 1:], step), dim=-1)
    return torch.cat(preds, dim=-1)


def _stepwise_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sqerr_sum: torch.Tensor,
    target_count: torch.Tensor,
    rel_sum: torch.Tensor,
    samples_by_step: torch.Tensor,
) -> None:
    rel_l2 = RelativeL2()
    for step in range(pred.shape[-1]):
        p = pred[..., step]
        y = target[..., step]
        diff = p - y
        sqerr_sum[step] += diff.pow(2).sum().detach().cpu()
        target_count[step] += diff.numel()
        rel = rel_l2(p, y)
        rel_sum[step] += rel.sum().detach().cpu()
        samples_by_step[step] += diff.shape[0]


def spectral_band_relative_l2(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    spatial_dim = pred.ndim - 2
    if spatial_dim not in {1, 2}:
        return {f"{name}_spectral_rel_l2": math.nan for name in DEFAULT_BANDS}

    if spatial_dim == 1:
        pred_fft = torch.fft.rfft(pred.float(), dim=1)
        target_fft = torch.fft.rfft(target.float(), dim=1)
        freqs = torch.fft.rfftfreq(pred.shape[1], device=pred.device).abs()
        radius = freqs / freqs.max().clamp_min(EPS)
    else:
        pred_fft = torch.fft.rfftn(pred.float(), dim=(1, 2))
        target_fft = torch.fft.rfftn(target.float(), dim=(1, 2))
        fx = torch.fft.fftfreq(pred.shape[1], device=pred.device).abs()
        fy = torch.fft.rfftfreq(pred.shape[2], device=pred.device).abs()
        radius = torch.sqrt(fx[:, None] ** 2 + fy[None, :] ** 2)
        radius = radius / radius.max().clamp_min(EPS)

    metrics = {}
    for name, (lo, hi) in DEFAULT_BANDS.items():
        mask = (radius >= lo) & (radius < hi)
        if spatial_dim == 1:
            mask = mask.view(1, mask.shape[0], 1).expand_as(pred_fft)
        else:
            mask = mask.view(1, mask.shape[0], mask.shape[1], 1).expand_as(pred_fft)
        diff = (pred_fft - target_fft)[mask]
        base = target_fft[mask]
        metrics[f"{name}_spectral_rel_l2"] = float(
            torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(base).clamp_min(EPS)
        )
    return metrics


def gradient_relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    spatial_dim = pred.ndim - 2
    if spatial_dim == 1:
        pred_grad = pred[:, 1:, :] - pred[:, :-1, :]
        target_grad = target[:, 1:, :] - target[:, :-1, :]
        diff = torch.cat(
            [
                (pred_grad - target_grad).reshape(pred.shape[0], -1),
            ],
            dim=1,
        )
        base = target_grad.reshape(target.shape[0], -1)
        return float(
            torch.linalg.vector_norm(diff, dim=1)
            .div(torch.linalg.vector_norm(base, dim=1).clamp_min(EPS))
            .mean()
        )
    if spatial_dim == 2:
        pred_dx = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, 1:, :, :] - target[:, :-1, :, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        diff = torch.cat(
            [
                (pred_dx - target_dx).reshape(pred.shape[0], -1),
                (pred_dy - target_dy).reshape(pred.shape[0], -1),
            ],
            dim=1,
        )
        base = torch.cat([target_dx.reshape(target.shape[0], -1), target_dy.reshape(target.shape[0], -1)], dim=1)
        return float(
            torch.linalg.vector_norm(diff, dim=1)
            .div(torch.linalg.vector_norm(base, dim=1).clamp_min(EPS))
            .mean()
        )
    return math.nan


def evaluate_official_kno_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    dataset: str,
    t_out: int,
    viscosity: str = "",
    run_name: str = "",
    model_name: str = "official_koopmanlab_kno",
    seed: int | str = "",
    epoch: int | str = "",
    warmup: int = 5,
    repeats: int = 20,
) -> dict[str, Any]:
    model.eval()
    mse = nn.MSELoss()
    rel_l2 = RelativeL2()
    sqerr_sum = torch.zeros(t_out)
    target_count = torch.zeros(t_out)
    rel_sum = torch.zeros(t_out)
    samples_by_step = torch.zeros(t_out)
    full_rel_sum = 0.0
    total_samples = 0
    spectral_sum = {f"{name}_spectral_rel_l2": 0.0 for name in DEFAULT_BANDS}
    gradient_sum = 0.0
    spectral_samples = 0
    first_pred = None
    first_target = None

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[:2]
            x = x.to(device)
            y = y.to(device)
            if dataset == "burgers":
                target = y.unsqueeze(-1)
                pred = predict_burgers(model, x)
            else:
                target = y
                pred = predict_rollout(model, x, t_out)
            _stepwise_stats(
                pred,
                target,
                sqerr_sum=sqerr_sum,
                target_count=target_count,
                rel_sum=rel_sum,
                samples_by_step=samples_by_step,
            )
            diff = pred - target
            full_rel = rel_l2(pred, target)
            full_rel_sum += float(full_rel.sum().item())
            total_samples += int(diff.shape[0])
            batch_size = int(diff.shape[0])
            batch_spectral = spectral_band_relative_l2(pred.detach(), target.detach())
            for key, value in batch_spectral.items():
                if not math.isnan(value):
                    spectral_sum[key] += float(value) * batch_size
            grad_value = gradient_relative_l2(pred.float(), target.float())
            if not math.isnan(grad_value):
                gradient_sum += float(grad_value) * batch_size
            spectral_samples += batch_size
            if first_pred is None:
                first_pred = pred.detach().cpu()
                first_target = target.detach().cpu()

    step_rel = rel_sum / samples_by_step.clamp_min(1)
    mse_by_step = sqerr_sum / target_count.clamp_min(1)
    steps = np.arange(t_out, dtype=np.float64)
    slope = 0.0 if t_out <= 1 else float(np.polyfit(steps, step_rel.numpy().astype(np.float64), 1)[0])

    params = count_params(model)
    peak_memory_gb, inference_ms_per_step, rollout_ms = measure_complexity(
        model,
        test_loader,
        device=device,
        dataset=dataset,
        t_out=t_out,
        warmup=warmup,
        repeats=repeats,
    )
    spectral = {}
    for key, value in spectral_sum.items():
        spectral[key] = value / max(spectral_samples, 1)
    spectral["gradient_rel_l2"] = gradient_sum / max(spectral_samples, 1)

    summary = {
        "run_name": run_name,
        "model": model_name,
        "dataset": dataset,
        "viscosity": viscosity,
        "seed": seed,
        "epoch": epoch,
        "test_mse": float(sqerr_sum.sum().item() / target_count.sum().clamp_min(1).item()),
        "step_rel_l2": float(step_rel.mean().item()),
        "full_rollout_rel_l2": full_rel_sum / max(total_samples, 1),
        "rollout_growth_slope": slope,
        "rollout_slope_fit_start": 0,
        "rollout_slope_fit_end": max(t_out - 1, 0),
        "params": params,
        "peak_memory_gb": peak_memory_gb,
        "inference_ms_per_step": inference_ms_per_step,
        "rollout_ms": rollout_ms,
        "spectral_band_config": DEFAULT_BANDS,
    }

    step_rows = [
        {
            "step": step,
            "mse": float(mse_by_step[step].item()),
            "rel_l2": float(step_rel[step].item()),
            "samples": int(samples_by_step[step].item()),
        }
        for step in range(t_out)
    ]

    return {
        "summary": summary,
        "step_rows": step_rows,
        "spectral_metrics": spectral,
        "first_pred": first_pred,
        "first_target": first_target,
    }


def measure_complexity(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    dataset: str,
    t_out: int,
    warmup: int,
    repeats: int,
) -> tuple[float, float, float]:
    model.eval()
    x, _ = next(iter(test_loader))
    x = x.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(warmup):
            if dataset == "burgers":
                _ = predict_burgers(model, x)
            else:
                _ = predict_rollout(model, x, t_out)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            if dataset == "burgers":
                _ = predict_burgers(model, x)
            else:
                _ = predict_rollout(model, x, t_out)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

    rollout_ms = elapsed * 1000.0 / max(repeats, 1)
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    return peak_memory_gb, rollout_ms / max(t_out, 1), rollout_ms


def save_stage0_outputs(
    out_dir: Path,
    result: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    write_json(out_dir / "evaluation_summary.json", summary)
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "run_name",
            "model",
            "dataset",
            "viscosity",
            "seed",
            "epoch",
            "test_mse",
            "step_rel_l2",
            "full_rollout_rel_l2",
            "rollout_growth_slope",
            "params",
            "peak_memory_gb",
            "inference_ms_per_step",
            "rollout_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: summary[key] for key in fields})
    with (out_dir / "rollout_error_by_step.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "mse", "rel_l2", "samples"])
        writer.writeheader()
        writer.writerows(result["step_rows"])
    with (out_dir / "complexity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["params", "peak_memory_gb", "inference_ms_per_step", "rollout_ms"])
        writer.writeheader()
        writer.writerow(
            {
                "params": summary["params"],
                "peak_memory_gb": summary["peak_memory_gb"],
                "inference_ms_per_step": summary["inference_ms_per_step"],
                "rollout_ms": summary["rollout_ms"],
            }
        )
    with (out_dir / "spectral_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in result["spectral_metrics"].items():
            writer.writerow({"metric": key, "value": f"{float(value):.8e}"})
        writer.writerow({"metric": "band_config_json", "value": json.dumps(DEFAULT_BANDS, sort_keys=True)})
