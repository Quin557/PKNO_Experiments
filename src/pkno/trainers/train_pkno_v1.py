"""Validation-safe curriculum training for Stage3_2 PKNO_v1."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer
from typing import Any

import torch
from torch import nn

from pkno.losses.pkno_v1_stability import growth_envelope_penalty, state_correction_penalty, temporal_state_penalty
from pkno.metrics.spectral import gradient_relative_l2, spectral_band_relative_l2
from pkno.models.pkno_v1 import PKNOV1Output
from pkno.trainers.train_rollout import CsvLogger, RelativeL2, count_params, write_env, write_json, write_rollout_error


@dataclass(frozen=True)
class PKNOV1TrainConfig:
    epochs: int = 500
    lr: float = 5e-4
    weight_decay: float = 1e-4
    step_size: int = 100
    gamma: float = 0.5
    pred_weight: float = 5.0
    recon_weight: float = 0.5
    state_weight: float = 1e-4
    smooth_weight: float = 1e-4
    growth_weight: float = 0.0
    growth_quantile: float = 0.99
    max_grad_norm: float | None = 1.0
    save_checkpoint: bool = False
    log_every: int = 1
    one_step_epochs: int = 50
    short_rollout_epochs: int = 50
    short_horizons: tuple[int, int] = (5, 10)
    resume: Path | None = None
    evaluate_test: bool = True


def _target_step(target: torch.Tensor, step: int) -> torch.Tensor:
    return target[..., step : step + 1]


def _append(history: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return pred if history.shape[-1] == 1 else torch.cat([history[..., 1:], pred], dim=-1)


def _teacher_forced_history(history: torch.Tensor, target: torch.Tensor, offset: int) -> torch.Tensor:
    if offset == 0:
        return history
    sequence = torch.cat([history, target[..., :offset]], dim=-1)
    return sequence[..., -history.shape[-1] :]


def curriculum_horizon(epoch: int, t_out: int, config: PKNOV1TrainConfig) -> tuple[int, bool]:
    if t_out == 1:
        return 1, True
    if epoch < config.one_step_epochs:
        return 1, True
    if epoch < config.one_step_epochs + config.short_rollout_epochs:
        midpoint = config.one_step_epochs + config.short_rollout_epochs // 2
        return min(config.short_horizons[0 if epoch < midpoint else 1], t_out), False
    return t_out, False


def estimate_growth_ceiling(loader: torch.utils.data.DataLoader, *, quantile: float, limit_batches: int = 32) -> float:
    ratios: list[torch.Tensor] = []
    for batch_index, (history, target, _) in enumerate(loader):
        sequence = torch.cat([history, target], dim=-1)
        dims = tuple(range(1, target.ndim))
        for step in range(target.shape[-1]):
            current = sequence[..., history.shape[-1] - 1 + step : history.shape[-1] + step]
            truth = sequence[..., history.shape[-1] + step : history.shape[-1] + step + 1]
            ratios.append(truth.square().mean(dim=dims).sqrt() / current.square().mean(dim=dims).sqrt().clamp_min(1e-12))
        if batch_index + 1 >= limit_batches:
            break
    if not ratios:
        return 1.0
    return float(torch.quantile(torch.cat(ratios), quantile).item())


def rollout_v1(
    model: nn.Module,
    history: torch.Tensor,
    target: torch.Tensor,
    condition: torch.Tensor,
    *,
    horizon: int,
    teacher_forced: bool,
    growth_ceiling: float | None,
) -> dict[str, torch.Tensor]:
    mse = nn.MSELoss()
    pred_steps: list[torch.Tensor] = []
    pred_mse = history.new_zeros(())
    recon_mse = history.new_zeros(())
    state_loss = history.new_zeros(())
    smooth_loss = history.new_zeros(())
    growth_loss = history.new_zeros(())
    step_rel = history.new_zeros(())
    rel_l2 = RelativeL2()
    previous_correction: torch.Tensor | None = None
    rollout_history = history

    for step in range(horizon):
        active_history = _teacher_forced_history(history, target, step) if teacher_forced else rollout_history
        out = model(active_history, condition)
        if not isinstance(out, PKNOV1Output):
            raise TypeError("PKNO_v1 model must return PKNOV1Output.")
        truth = _target_step(target, step)
        pred_mse = pred_mse + mse(out.prediction, truth)
        recon_mse = recon_mse + mse(out.reconstruction, active_history)
        state_loss = state_loss + state_correction_penalty(out.state_correction)
        smooth_loss = smooth_loss + temporal_state_penalty(out.state_correction, previous_correction)
        if growth_ceiling is not None:
            growth_loss = growth_loss + growth_envelope_penalty(out.prediction, active_history[..., -1:], growth_ceiling)
        step_rel = step_rel + rel_l2(out.prediction, truth)
        pred_steps.append(out.prediction)
        previous_correction = out.state_correction.detach()
        rollout_history = _append(rollout_history, out.prediction)

    prediction = torch.cat(pred_steps, dim=-1)
    return {
        "prediction": prediction,
        "pred_mse": pred_mse / horizon,
        "recon_mse": recon_mse / horizon,
        "state_loss": state_loss / horizon,
        "smooth_loss": smooth_loss / horizon,
        "growth_loss": growth_loss / horizon,
        "step_rel": step_rel / horizon,
    }


def evaluate_v1(model: nn.Module, loader: torch.utils.data.DataLoader, *, device: torch.device, t_out: int) -> dict[str, Any]:
    model.eval()
    rel_l2 = RelativeL2()
    mse = nn.MSELoss()
    stats: dict[str, Any] = {"samples": 0, "full_rel_sum": 0.0, "step_rel_sum": 0.0, "pred_mse_sum": 0.0, "step_mse_sum": torch.zeros(t_out), "first_pred": None, "first_target": None}
    with torch.no_grad():
        for history, target, condition in loader:
            history, target, condition = history.to(device), target.to(device), condition.to(device)
            out = rollout_v1(model, history, target, condition, horizon=t_out, teacher_forced=False, growth_ceiling=None)
            prediction = out["prediction"]
            batch = history.shape[0]
            stats["samples"] += batch
            stats["full_rel_sum"] += float(rel_l2(prediction, target).item())
            stats["step_rel_sum"] += float(out["step_rel"].item())
            stats["pred_mse_sum"] += float(out["pred_mse"].item()) * batch
            for step in range(t_out):
                stats["step_mse_sum"][step] += float(mse(_target_step(prediction, step), _target_step(target, step)).item()) * batch
            if stats["first_pred"] is None:
                stats["first_pred"] = prediction.detach().cpu()
                stats["first_target"] = target.detach().cpu()
    samples = max(stats["samples"], 1)
    stats["full_rel_l2"] = stats["full_rel_sum"] / samples
    stats["step_rel_l2"] = stats["step_rel_sum"] / samples
    stats["pred_mse"] = stats["pred_mse_sum"] / samples
    stats["step_mse"] = stats["step_mse_sum"] / samples
    return stats


def _write_spectral(path: Path, pred: torch.Tensor, target: torch.Tensor) -> None:
    metrics = spectral_band_relative_l2(pred.float(), target.float())
    metrics["gradient_rel_l2"] = gradient_relative_l2(pred.float(), target.float())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": f"{float(value.item()):.8e}"})


def train_pkno_v1(
    model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader | None,
    test_loader: torch.utils.data.DataLoader, *, out_dir: Path, device: torch.device,
    config: PKNOV1TrainConfig, t_out: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    params = count_params(model)
    write_env(out_dir / "env.txt", params)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    start_epoch = 0
    if config.resume is not None:
        checkpoint = torch.load(config.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
    ceiling = estimate_growth_ceiling(train_loader, quantile=config.growth_quantile) if config.growth_weight > 0 else None
    write_json(out_dir / "stability_config.json", {"growth_ceiling": ceiling, "growth_quantile": config.growth_quantile, "state_weight": config.state_weight, "smooth_weight": config.smooth_weight, "growth_weight": config.growth_weight})
    logger = CsvLogger(out_dir / "metrics.csv", ["epoch", "seconds", "train_horizon", "train_loss", "train_pred_mse", "train_full_rel_l2", "val_full_rel_l2", "val_pred_mse", "lr", "params"])
    best = float("inf")
    try:
        for epoch in range(start_epoch, config.epochs):
            model.train()
            started = default_timer()
            horizon, teacher_forced = curriculum_horizon(epoch, t_out, config)
            total_loss = total_mse = total_rel = 0.0
            samples = 0
            for history, target, condition in train_loader:
                history, target, condition = history.to(device), target.to(device), condition.to(device)
                out = rollout_v1(model, history, target, condition, horizon=horizon, teacher_forced=teacher_forced, growth_ceiling=ceiling)
                loss = (config.pred_weight * out["pred_mse"] + config.recon_weight * out["recon_mse"] + config.state_weight * out["state_loss"] + config.smooth_weight * out["smooth_loss"] + config.growth_weight * out["growth_loss"])
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(f"Non-finite PKNO_v1 loss at epoch={epoch}.")
                optimizer.zero_grad()
                loss.backward()
                if config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                batch = history.shape[0]
                samples += batch
                total_loss += float(loss.item()) * batch
                total_mse += float(out["pred_mse"].item()) * batch
                total_rel += float(RelativeL2()(out["prediction"], target[..., :horizon]).item())
            val = evaluate_v1(model, val_loader, device=device, t_out=t_out) if val_loader is not None else None
            scheduler.step()
            seconds = default_timer() - started
            row = {"epoch": epoch, "seconds": f"{seconds:.6f}", "train_horizon": horizon, "train_loss": f"{total_loss / max(samples, 1):.8e}", "train_pred_mse": f"{total_mse / max(samples, 1):.8e}", "train_full_rel_l2": f"{total_rel / max(samples, 1):.8e}", "val_full_rel_l2": "" if val is None else f"{val['full_rel_l2']:.8e}", "val_pred_mse": "" if val is None else f"{val['pred_mse']:.8e}", "lr": f"{scheduler.get_last_lr()[0]:.8e}", "params": params}
            logger.write(row)
            if val is not None and config.save_checkpoint and val["full_rel_l2"] < best:
                best = val["full_rel_l2"]
                torch.save(model.state_dict(), out_dir / "checkpoint_best_val.pt")
            if config.save_checkpoint:
                torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, out_dir / "checkpoint_last.pt")
            if epoch % config.log_every == 0:
                print(f"epoch {epoch:04d} | {seconds:.2f}s | horizon {horizon:02d} | train {float(row['train_full_rel_l2']):.6e} | val {row['val_full_rel_l2'] or 'n/a'}")
    finally:
        logger.close()
    torch.save(model.state_dict(), out_dir / "checkpoint_final.pt")
    if not config.evaluate_test:
        return
    test = evaluate_v1(model, test_loader, device=device, t_out=t_out)
    write_rollout_error(out_dir / "rollout_error_by_step.csv", test["step_mse"])
    if test["first_pred"] is not None:
        _write_spectral(out_dir / "spectral_metrics.csv", test["first_pred"], test["first_target"])
    write_json(out_dir / "evaluation_summary.json", {"split": "test", "checkpoint": "final", "full_rel_l2": test["full_rel_l2"], "step_rel_l2": test["step_rel_l2"], "pred_mse": test["pred_mse"], "params": params, "growth_ceiling": ceiling})
