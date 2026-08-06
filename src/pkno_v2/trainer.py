"""Memory-conscious autoregressive trainer for PKNO_v2-A."""

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
from pkno.trainers.train_rollout import RelativeL2, count_params

from .losses import gradient_mse, high_frequency_mse, late_step_weights
from .model import PKNOV2Output


@dataclass(frozen=True)
class V2TrainConfig:
    epochs: int = 500
    lr: float = 5e-4
    weight_decay: float = 1e-4
    step_size: int = 100
    gamma: float = 0.5
    recon_weight: float = 0.5
    gradient_weight: float = 1e-3
    spectral_weight: float = 1e-4
    gate_weight: float = 1e-4
    late_final_weight: float = 2.0
    max_grad_norm: float = 1.0
    save_checkpoint: bool = False
    log_every: int = 1


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _target_step(target: torch.Tensor, step: int) -> torch.Tensor:
    return target[..., step : step + 1]


def _append(history: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return torch.cat([history[..., 1:], pred], dim=-1) if history.shape[-1] > 1 else pred


def _horizon(epoch: int, t_out: int) -> tuple[int, bool]:
    if t_out == 1:
        return 1, True
    if epoch < 40:
        return 1, True
    if epoch < 80:
        return min(5, t_out), False
    if epoch < 120:
        return min(10, t_out), False
    return t_out, False


def rollout(model: nn.Module, history: torch.Tensor, target: torch.Tensor, condition: torch.Tensor,
            horizon: int, teacher_forced: bool, cfg: V2TrainConfig, compute_aux: bool = True) -> dict[str, torch.Tensor]:
    mse = nn.MSELoss()
    weights = late_step_weights(horizon, cfg.late_final_weight, history.device)
    pred_steps: list[torch.Tensor] = []
    pred_loss = history.new_zeros(()); recon_loss = history.new_zeros(())
    grad_loss = history.new_zeros(()); spec_loss = history.new_zeros(()); gate_loss = history.new_zeros(())
    rollout_history = history
    for step in range(horizon):
        active = torch.cat([history, target[..., :step]], dim=-1)[..., -history.shape[-1]:] if teacher_forced and step else rollout_history
        out = model(active, condition)
        if not isinstance(out, PKNOV2Output):
            raise TypeError("PKNO_v2 model must return PKNOV2Output")
        truth = _target_step(target, step)
        pred_loss = pred_loss + weights[step] * mse(out.prediction, truth)
        recon_loss = recon_loss + mse(out.reconstruction, active)
        if compute_aux:
            grad_loss = grad_loss + gradient_mse(out.prediction, truth)
            spec_loss = spec_loss + high_frequency_mse(out.prediction, truth)
        gate_loss = gate_loss + out.state_gate.square().mean() + (out.eta / max(1, model.decompose)).square()
        pred_steps.append(out.prediction)
        rollout_history = _append(rollout_history, out.prediction)
    prediction = torch.cat(pred_steps, dim=-1)
    return {"prediction": prediction, "pred_loss": pred_loss / weights.sum().clamp_min(1e-12), "recon_loss": recon_loss / horizon,
            "gradient_loss": grad_loss / horizon, "spectral_loss": spec_loss / horizon, "gate_loss": gate_loss / horizon}


def _evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, t_out: int) -> dict[str, Any]:
    model.eval(); metric = RelativeL2(); mse = nn.MSELoss()
    full = step = pred_mse = 0.0; samples = 0; step_mse = torch.zeros(t_out)
    first_pred = first_target = None
    with torch.no_grad():
        for history, target, condition in loader:
            history, target, condition = history.to(device), target.to(device), condition.to(device)
            out = rollout(model, history, target, condition, t_out, False, V2TrainConfig(), compute_aux=False)
            pred = out["prediction"]; batch = history.shape[0]
            full += float(metric(pred, target).item()); step += float(sum(metric(pred[..., i:i+1], target[..., i:i+1]).item() for i in range(t_out))); pred_mse += float(nn.functional.mse_loss(pred, target).item()) * batch; samples += batch
            for i in range(t_out): step_mse[i] += float(mse(pred[..., i:i+1], target[..., i:i+1]).item()) * batch
            if first_pred is None: first_pred, first_target = pred.cpu(), target.cpu()
    return {"full_rel_l2": full / max(samples, 1), "step_rel_l2": step / max(samples * t_out, 1), "pred_mse": pred_mse / max(samples, 1), "step_mse": step_mse / max(samples, 1), "first_pred": first_pred, "first_target": first_target}


def train_v2(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader | None,
             test_loader: torch.utils.data.DataLoader, *, out_dir: Path, device: torch.device, config: V2TrainConfig, t_out: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True); model.to(device)
    params = count_params(model)
    (out_dir / "env.txt").write_text("\n".join([f"python={sys.version}", f"platform={platform.platform()}", f"torch={torch.__version__}", f"cuda={torch.cuda.is_available()}", f"params={params}", f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}" ]) + "\n", encoding="utf-8")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.step_size, config.gamma)
    fields = ["epoch", "seconds", "horizon", "train_loss", "train_pred_mse", "val_full_rel_l2", "test_full_rel_l2", "lr", "params"]
    handle = (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8"); logger = csv.DictWriter(handle, fieldnames=fields); logger.writeheader(); best = float("inf")
    try:
        for epoch in range(config.epochs):
            model.train(); started = default_timer(); horizon, forced = _horizon(epoch, t_out); total = pred_total = 0.0; seen = 0
            for history, target, condition in train_loader:
                history, target, condition = history.to(device), target.to(device), condition.to(device)
                out = rollout(model, history, target, condition, horizon, forced, config)
                loss = out["pred_loss"] + config.recon_weight * out["recon_loss"] + config.gradient_weight * out["gradient_loss"] + config.spectral_weight * out["spectral_loss"] + config.gate_weight * out["gate_loss"]
                if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite PKNO_v2 loss at epoch={epoch}")
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm); optimizer.step()
                batch = history.shape[0]; seen += batch; total += float(loss.item()) * batch; pred_total += float(out["pred_loss"].item()) * batch
            val = _evaluate(model, val_loader, device, t_out) if val_loader is not None else None
            test = _evaluate(model, test_loader, device, t_out)
            scheduler.step(); seconds = default_timer() - started
            row = {"epoch": epoch, "seconds": f"{seconds:.6f}", "horizon": horizon, "train_loss": f"{total/max(seen,1):.8e}", "train_pred_mse": f"{pred_total/max(seen,1):.8e}", "val_full_rel_l2": "" if val is None else f"{val['full_rel_l2']:.8e}", "test_full_rel_l2": f"{test['full_rel_l2']:.8e}", "lr": f"{scheduler.get_last_lr()[0]:.8e}", "params": params}; logger.writerow(row); handle.flush()
            score = val["full_rel_l2"] if val is not None else test["full_rel_l2"]
            if config.save_checkpoint and score < best:
                best = score; torch.save({"model": model.state_dict(), "epoch": epoch, "score": score}, out_dir / "checkpoint_best.pt")
            if epoch % config.log_every == 0: print(f"epoch {epoch:04d} | {seconds:.2f}s | horizon {horizon:02d} | train {float(row['train_loss']):.6e} | test {float(row['test_full_rel_l2']):.6e}")
    finally:
        handle.close()
    final = _evaluate(model, test_loader, device, t_out)
    with (out_dir / "rollout_error_by_step.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "mse"]); writer.writeheader()
        for i, v in enumerate(final["step_mse"].tolist()): writer.writerow({"step": i, "mse": f"{v:.8e}"})
    if final["first_pred"] is not None:
        metrics = spectral_band_relative_l2(final["first_pred"].float(), final["first_target"].float()); metrics["gradient_rel_l2"] = gradient_relative_l2(final["first_pred"].float(), final["first_target"].float())
        _write_json(out_dir / "spectral_metrics.json", {k: float(v.item()) for k, v in metrics.items()})
    if config.save_checkpoint: torch.save({"model": model.state_dict(), "epoch": config.epochs - 1}, out_dir / "checkpoint_last.pt")
