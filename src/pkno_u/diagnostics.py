"""Post-training stability diagnostics for PKNO-U rollouts."""

from __future__ import annotations

import csv
from pathlib import Path

import torch


def _append_history(history: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    if history.shape[-1] == prediction.shape[-1]:
        return prediction
    return torch.cat([history[..., 1:], prediction], dim=-1)


def collect_rollout_diagnostics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    t_out: int,
    max_batches: int = 1,
) -> list[dict[str, float]]:
    """Collect per-step model diagnostics on a bounded number of evaluation batches."""

    model.eval()
    totals: list[dict[str, float]] = [dict() for _ in range(t_out)]
    counts = [0 for _ in range(t_out)]
    with torch.no_grad():
        for batch_index, (history, _, condition) in enumerate(loader):
            if batch_index >= max_batches:
                break
            history = history.to(device)
            condition = condition.to(device)
            for step in range(t_out):
                prediction, _ = model(history, condition)
                diagnostics = getattr(model, "last_diagnostics", {})
                for name, value in diagnostics.items():
                    totals[step][name] = totals[step].get(name, 0.0) + float(value.item())
                counts[step] += 1
                history = _append_history(history, prediction)
    rows: list[dict[str, float]] = []
    for step, values in enumerate(totals):
        row: dict[str, float] = {"step": float(step)}
        for name, value in values.items():
            row[name] = value / max(counts[step], 1)
        rows.append(row)
    return rows


def write_rollout_diagnostics(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ["step", "condition_gate", "matrix_spectral_mean", "matrix_spectral_max", "latent_rms", "unet_highpass_rms"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, 0.0) for name in fieldnames})
