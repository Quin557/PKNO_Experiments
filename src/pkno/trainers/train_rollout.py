"""Autoregressive rollout training utilities for Stage 3."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer
from typing import Any

import torch
from torch import nn

from pkno.metrics.spectral import gradient_relative_l2, spectral_band_relative_l2


def count_params(model: nn.Module) -> int:
    """Count real parameters, treating complex tensors as two real tensors."""

    total = 0
    for param in model.parameters():
        total += param.numel() * (2 if param.is_complex() else 1)
    return total


class RelativeL2:
    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch = pred.shape[0]
        diff = torch.linalg.vector_norm((pred - target).reshape(batch, -1), dim=1)
        denom = torch.linalg.vector_norm(target.reshape(batch, -1), dim=1).clamp_min(self.eps)
        return (diff / denom).sum()


class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


@dataclass(frozen=True)
class RolloutTrainConfig:
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    step_size: int = 100
    gamma: float = 0.5
    pred_weight: float = 5.0
    recon_weight: float = 0.5
    max_grad_norm: float | None = None
    save_checkpoint: bool = False
    log_every: int = 1


def write_json(path: Path, obj: dict[str, Any]) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    path.write_text(json.dumps(convert(obj), indent=2, sort_keys=True), encoding="utf-8")


def write_env(path: Path, params: int) -> None:
    lines = [
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_version={torch.version.cuda}",
        f"cuda_device_count={torch.cuda.device_count()}",
        f"params_count_complex_as_2={params}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
    ]
    if torch.cuda.is_available():
        lines.append(f"cuda_device_name={torch.cuda.get_device_name(0)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _target_step(target: torch.Tensor, step: int, output_channels: int) -> torch.Tensor:
    if output_channels == 1:
        if target.shape[-1] <= step:
            raise ValueError(
                f"Target has {target.shape[-1]} rollout channels, cannot read step {step}."
            )
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
    condition: torch.Tensor,
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
        pred_step, recon = model(history, condition)
        truth_step = _target_step(target, step, output_channels)
        pred_mse = pred_mse + mse(pred_step, truth_step)
        recon_mse = recon_mse + mse(recon, history)
        step_rel = step_rel + rel_l2(pred_step, truth_step)
        pred_steps.append(pred_step if output_channels == 1 else pred_step.unsqueeze(-2))
        history = _append_prediction(history, pred_step)

    pred = torch.cat(pred_steps, dim=-1 if output_channels == 1 else -2)
    return {
        "pred": pred,
        "pred_mse": pred_mse,
        "recon_mse": recon_mse,
        "step_rel": step_rel,
    }


def evaluate_rollout(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    t_out: int,
    output_channels: int,
) -> dict[str, Any]:
    model.eval()
    mse = nn.MSELoss()
    rel_l2 = RelativeL2()
    stats: dict[str, Any] = {
        "samples": 0,
        "step_rel_sum": 0.0,
        "full_rel_sum": 0.0,
        "pred_mse_sum": 0.0,
        "recon_mse_sum": 0.0,
        "step_mse_sum": torch.zeros(t_out),
        "first_pred": None,
        "first_target": None,
    }

    with torch.no_grad():
        for history, target, condition in loader:
            history = history.to(device)
            target = target.to(device)
            condition = condition.to(device)
            batch = history.shape[0]
            out = rollout_model(
                model,
                history,
                target,
                condition,
                t_out=t_out,
                output_channels=output_channels,
                mse=mse,
                rel_l2=rel_l2,
            )
            pred = out["pred"]
            stats["samples"] += batch
            stats["step_rel_sum"] += float(out["step_rel"].item())
            stats["full_rel_sum"] += float(rel_l2(pred, target).item())
            stats["pred_mse_sum"] += float(out["pred_mse"].item())
            stats["recon_mse_sum"] += float(out["recon_mse"].item() / t_out)
            for step in range(t_out):
                truth_step = _target_step(target, step, output_channels)
                pred_step = _target_step(pred, step, output_channels)
                stats["step_mse_sum"][step] += float(mse(pred_step, truth_step).item()) * batch
            if stats["first_pred"] is None:
                stats["first_pred"] = pred.detach().cpu()
                stats["first_target"] = target.detach().cpu()

    samples = max(stats["samples"], 1)
    stats["step_rel_l2"] = stats["step_rel_sum"] / samples / t_out
    stats["full_rel_l2"] = stats["full_rel_sum"] / samples
    stats["pred_mse"] = stats["pred_mse_sum"] / max(len(loader), 1)
    stats["recon_mse"] = stats["recon_mse_sum"] / max(len(loader), 1)
    stats["step_mse"] = stats["step_mse_sum"] / samples
    return stats


def write_rollout_error(path: Path, step_mse: torch.Tensor) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "mse"])
        writer.writeheader()
        for step, value in enumerate(step_mse.reshape(-1).tolist()):
            writer.writerow({"step": step, "mse": f"{value:.8e}"})


def write_spectral_metrics(path: Path, pred: torch.Tensor, target: torch.Tensor) -> None:
    pred = pred.float()
    target = target.float()
    metrics = spectral_band_relative_l2(pred, target)
    metrics["gradient_rel_l2"] = gradient_relative_l2(pred, target)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": f"{float(value.item()):.8e}"})


def train_autoregressive(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    *,
    out_dir: Path,
    device: torch.device,
    config: RolloutTrainConfig,
    t_out: int,
    output_channels: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    params = count_params(model)
    write_env(out_dir / "env.txt", params)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    mse = nn.MSELoss()
    rel_l2 = RelativeL2()
    logger = CsvLogger(
        out_dir / "metrics.csv",
        [
            "epoch",
            "seconds",
            "train_step_rel_l2",
            "train_full_rel_l2",
            "test_step_rel_l2",
            "test_full_rel_l2",
            "train_pred_mse",
            "train_recon_mse",
            "test_pred_mse",
            "test_recon_mse",
            "params",
            "lr",
        ],
    )
    best = float("inf")

    try:
        for epoch in range(config.epochs):
            model.train()
            started = default_timer()
            train_stats = {
                "samples": 0,
                "step_rel_sum": 0.0,
                "full_rel_sum": 0.0,
                "pred_mse_sum": 0.0,
                "recon_mse_sum": 0.0,
            }
            for history, target, condition in train_loader:
                history = history.to(device)
                target = target.to(device)
                condition = condition.to(device)
                batch = history.shape[0]
                out = rollout_model(
                    model,
                    history,
                    target,
                    condition,
                    t_out=t_out,
                    output_channels=output_channels,
                    mse=mse,
                    rel_l2=rel_l2,
                )
                pred = out["pred"]
                loss = config.pred_weight * out["pred_mse"] + config.recon_weight * out["recon_mse"]
                optimizer.zero_grad()
                loss.backward()
                if config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                train_stats["samples"] += batch
                train_stats["step_rel_sum"] += float(out["step_rel"].item())
                train_stats["full_rel_sum"] += float(rel_l2(pred, target).item())
                train_stats["pred_mse_sum"] += float(out["pred_mse"].item())
                train_stats["recon_mse_sum"] += float(out["recon_mse"].item() / t_out)

            test_stats = evaluate_rollout(
                model,
                test_loader,
                device=device,
                t_out=t_out,
                output_channels=output_channels,
            )
            scheduler.step()
            seconds = default_timer() - started
            train_samples = max(train_stats["samples"], 1)
            row = {
                "epoch": epoch,
                "seconds": f"{seconds:.6f}",
                "train_step_rel_l2": f"{train_stats['step_rel_sum'] / train_samples / t_out:.8e}",
                "train_full_rel_l2": f"{train_stats['full_rel_sum'] / train_samples:.8e}",
                "test_step_rel_l2": f"{test_stats['step_rel_l2']:.8e}",
                "test_full_rel_l2": f"{test_stats['full_rel_l2']:.8e}",
                "train_pred_mse": f"{train_stats['pred_mse_sum'] / max(len(train_loader), 1):.8e}",
                "train_recon_mse": f"{train_stats['recon_mse_sum'] / max(len(train_loader), 1):.8e}",
                "test_pred_mse": f"{test_stats['pred_mse']:.8e}",
                "test_recon_mse": f"{test_stats['recon_mse']:.8e}",
                "params": params,
                "lr": f"{scheduler.get_last_lr()[0]:.8e}",
            }
            logger.write(row)

            score = float(row["test_full_rel_l2"])
            if config.save_checkpoint and score < best:
                best = score
                torch.save(model.state_dict(), out_dir / "checkpoint_best.pt")
            if epoch % config.log_every == 0:
                print(
                    f"epoch {epoch:04d} | {seconds:.2f}s | "
                    f"train_full {float(row['train_full_rel_l2']):.6e} | "
                    f"test_full {float(row['test_full_rel_l2']):.6e}"
                )
    finally:
        logger.close()

    final_stats = evaluate_rollout(
        model,
        test_loader,
        device=device,
        t_out=t_out,
        output_channels=output_channels,
    )
    write_rollout_error(out_dir / "rollout_error_by_step.csv", final_stats["step_mse"])
    if final_stats["first_pred"] is not None and final_stats["first_target"] is not None:
        write_spectral_metrics(
            out_dir / "spectral_metrics.csv",
            final_stats["first_pred"],
            final_stats["first_target"],
        )
    if config.save_checkpoint:
        torch.save(model.state_dict(), out_dir / "checkpoint_last.pt")
