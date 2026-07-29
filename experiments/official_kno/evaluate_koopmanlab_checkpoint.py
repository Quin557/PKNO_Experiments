from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from train_koopmanlab_ns import build_navier_stokes_loaders
from train_koopmanlab_shallow_water import build_shallow_water_loaders


EPS = 1e-12
DEFAULT_BANDS = {
    "low": [0.0, 0.33],
    "mid": [0.33, 0.66],
    "high": [0.66, 1.01],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate a Stage 0 official KNO checkpoint")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage0_kno_baseline_eval"))
    parser.add_argument("--dataset", choices=["auto", "burgers", "navier_stokes", "shallow_water"], default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


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


def write_env(path: Path, params: int) -> None:
    lines = [
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_version={torch.version.cuda}",
        f"cuda_device_count={torch.cuda.device_count()}",
        f"params={params}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
    ]
    if torch.cuda.is_available():
        lines.append(f"cuda_device_name={torch.cuda.get_device_name(0)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_run_args(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "args.json"
    if not path.exists():
        raise FileNotFoundError(f"args.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def infer_dataset(run_args: dict[str, Any], run_name: str, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    name = run_name.lower()
    if "viscosity_type" in run_args or "ns_" in name or "navier" in name:
        return "navier_stokes"
    if "shallow" in name:
        return "shallow_water"
    if "burgers" in name:
        return "burgers"
    raise ValueError(f"Cannot infer dataset for run: {run_name}")


def namespace_from_args(run_args: dict[str, Any]) -> argparse.Namespace:
    converted = {}
    for key, value in run_args.items():
        converted[key.replace("-", "_")] = value
    for key in ["koopmanlab_root", "data_path", "output_dir"]:
        if key in converted:
            converted[key] = Path(converted[key])
    return argparse.Namespace(**converted)


def resolve_device(name: str | None) -> torch.device:
    if name is None:
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def count_params(model: torch.nn.Module) -> int:
    return sum(param.numel() * (2 if param.is_complex() else 1) for param in model.parameters())


def build_loaders_and_model(dataset: str, run_args: dict[str, Any], device: torch.device):
    args = namespace_from_args(run_args)
    koopmanlab_root = Path(run_args.get("koopmanlab_root", "external/KoopmanLab"))
    if not koopmanlab_root.exists():
        raise FileNotFoundError(f"KoopmanLab root not found: {koopmanlab_root}")
    sys.path.insert(0, str(koopmanlab_root.resolve()))
    import koopmanlab as kp  # noqa: PLC0415

    if dataset == "burgers":
        train_loader, test_loader = kp.data.burgers(
            str(args.data_path),
            batch_size=args.batch_size,
            sub=args.sub,
        )
        model = kp.model.koopman(
            backbone="KNO1d",
            autoencoder="MLP",
            o=args.operator_size,
            m=args.modes,
            r=args.decompose,
            t_in=1,
            device=device,
        )
        t_out = 1
    elif dataset == "navier_stokes":
        train_loader, test_loader = build_navier_stokes_loaders(args, kp)
        model = kp.model.koopman(
            backbone="KNO2d",
            autoencoder="MLP",
            o=args.operator_size,
            m=args.modes,
            r=args.decompose,
            t_in=args.t_in,
            device=device,
        )
        t_out = args.t_out
    elif dataset == "shallow_water":
        train_loader, test_loader = build_shallow_water_loaders(args)
        model = kp.model.koopman(
            backbone="KNO2d",
            autoencoder="MLP",
            o=args.operator_size,
            m=args.modes,
            r=args.decompose,
            t_in=args.t_in,
            device=device,
        )
        t_out = args.t_out
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    model.compile()
    model.opt_init(
        "Adam",
        lr=float(run_args.get("lr", 0.001)),
        step_size=int(run_args.get("step_size", 100)),
        gamma=float(run_args.get("gamma", 0.5)),
    )
    return train_loader, test_loader, model, t_out


def load_model_checkpoint(model: Any, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Not a valid checkpoint with model_state_dict: {checkpoint_path}")
    model.kernel.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def predict_burgers(model: Any, x: torch.Tensor) -> torch.Tensor:
    pred, _ = model.kernel(x)
    return pred.unsqueeze(-1)


def predict_rollout(model: Any, x: torch.Tensor, t_out: int) -> torch.Tensor:
    preds = []
    history = x
    for _ in range(t_out):
        im, _ = model.kernel(history)
        step = im[..., -1:]
        preds.append(step)
        history = torch.cat((history[..., 1:], step), dim=-1)
    return torch.cat(preds, dim=-1)


def add_step_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    sqerr_sum: torch.Tensor,
    target_count: torch.Tensor,
    rel_sum: torch.Tensor,
    samples_by_step: torch.Tensor,
) -> None:
    for step in range(pred.shape[-1]):
        p = pred[..., step]
        y = target[..., step]
        diff = p - y
        sqerr_sum[step] += diff.pow(2).sum().detach().cpu()
        target_count[step] += diff.numel()
        diff_norm = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=1)
        target_norm = torch.linalg.vector_norm(y.reshape(y.shape[0], -1), dim=1).clamp_min(EPS)
        rel = diff_norm / target_norm
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
        return float(
            torch.linalg.vector_norm((pred_grad - target_grad).reshape(pred.shape[0], -1), dim=1)
            .div(torch.linalg.vector_norm(target_grad.reshape(target.shape[0], -1), dim=1).clamp_min(EPS))
            .mean()
        )
    if spatial_dim == 2:
        pred_dx = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, 1:, :, :] - target[:, :-1, :, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        diff = torch.cat([(pred_dx - target_dx).reshape(pred.shape[0], -1), (pred_dy - target_dy).reshape(pred.shape[0], -1)], dim=1)
        base = torch.cat([target_dx.reshape(target.shape[0], -1), target_dy.reshape(target.shape[0], -1)], dim=1)
        return float(
            torch.linalg.vector_norm(diff, dim=1)
            .div(torch.linalg.vector_norm(base, dim=1).clamp_min(EPS))
            .mean()
        )
    return math.nan


def evaluate_full_test(model: Any, test_loader: torch.utils.data.DataLoader, device: torch.device, t_out: int, dataset: str) -> dict[str, Any]:
    model.kernel.eval()
    sqerr_sum = torch.zeros(t_out)
    target_count = torch.zeros(t_out)
    rel_sum = torch.zeros(t_out)
    samples_by_step = torch.zeros(t_out)
    full_rel_sum = 0.0
    samples = 0
    first_pred = None
    first_target = None

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            if dataset == "burgers":
                target = y.unsqueeze(-1)
                pred = predict_burgers(model, x)
            else:
                target = y
                pred = predict_rollout(model, x, t_out)
            add_step_stats(
                pred,
                target,
                sqerr_sum=sqerr_sum,
                target_count=target_count,
                rel_sum=rel_sum,
                samples_by_step=samples_by_step,
            )
            diff = pred - target
            full_rel = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=1) / torch.linalg.vector_norm(
                target.reshape(target.shape[0], -1),
                dim=1,
            ).clamp_min(EPS)
            full_rel_sum += float(full_rel.sum().item())
            samples += int(diff.shape[0])
            if first_pred is None:
                first_pred = pred.detach().cpu()
                first_target = target.detach().cpu()

    mse_by_step = sqerr_sum / target_count.clamp_min(1)
    rel_by_step = rel_sum / samples_by_step.clamp_min(1)
    steps = np.arange(t_out, dtype=np.float64)
    slope = 0.0 if t_out == 1 else float(np.polyfit(steps, rel_by_step.numpy().astype(np.float64), 1)[0])
    return {
        "test_mse": float(sqerr_sum.sum().item() / target_count.sum().clamp_min(1).item()),
        "step_rel_l2": float(rel_by_step.mean().item()),
        "full_rollout_rel_l2": full_rel_sum / max(samples, 1),
        "rollout_growth_slope": slope,
        "rollout_slope_fit_start": 0,
        "rollout_slope_fit_end": t_out - 1,
        "samples": samples,
        "mse_by_step": mse_by_step,
        "rel_by_step": rel_by_step,
        "samples_by_step": samples_by_step,
        "first_pred": first_pred,
        "first_target": first_target,
    }


def measure_complexity(model: Any, test_loader: torch.utils.data.DataLoader, device: torch.device, t_out: int, dataset: str, warmup: int, repeats: int) -> dict[str, float]:
    model.kernel.eval()
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
    peak_memory_gb = 0.0
    if device.type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    return {
        "peak_memory_gb": peak_memory_gb,
        "inference_ms_per_step": rollout_ms / max(t_out, 1),
        "rollout_ms": rollout_ms,
    }


def write_rollout_error(path: Path, result: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "mse", "rel_l2", "samples"])
        writer.writeheader()
        for step in range(len(result["mse_by_step"])):
            writer.writerow(
                {
                    "step": step,
                    "mse": f"{float(result['mse_by_step'][step]):.8e}",
                    "rel_l2": f"{float(result['rel_by_step'][step]):.8e}",
                    "samples": int(result["samples_by_step"][step].item()),
                }
            )


def write_spectral(path: Path, pred: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    metrics = spectral_band_relative_l2(pred, target)
    metrics["gradient_rel_l2"] = gradient_relative_l2(pred.float(), target.float())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": f"{float(value):.8e}"})
        writer.writerow({"metric": "band_config_json", "value": json.dumps(DEFAULT_BANDS, sort_keys=True)})
    return metrics


def main() -> None:
    args = parse_args()
    run_args = load_run_args(args.run_dir)
    run_name = str(run_args.get("run_name", args.run_dir.name))
    checkpoint_path = args.checkpoint or args.run_dir / "checkpoint_last.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    dataset = infer_dataset(run_args, run_name, args.dataset)
    device = resolve_device(args.device or run_args.get("device_resolved") or run_args.get("device"))
    eval_run_name = args.eval_run_name or f"{run_name}_eval_v1"
    out_dir = args.output_dir / eval_run_name
    if out_dir.exists():
        raise FileExistsError(f"Evaluation output already exists, choose another --eval-run-name: {out_dir}")
    out_dir.mkdir(parents=True)

    _, test_loader, model, t_out = build_loaders_and_model(dataset, run_args, device)
    checkpoint = load_model_checkpoint(model, checkpoint_path, device)
    params = count_params(model.kernel)
    write_env(out_dir / "env.txt", params)

    result = evaluate_full_test(model, test_loader, device, t_out, dataset)
    complexity = measure_complexity(model, test_loader, device, t_out, dataset, args.warmup, args.repeats)
    spectral = write_spectral(out_dir / "spectral_metrics.csv", result["first_pred"], result["first_target"])

    summary = {
        "run_name": eval_run_name,
        "source_run_name": run_name,
        "model": "official_koopmanlab_kno",
        "dataset": dataset,
        "viscosity": run_args.get("viscosity_type", ""),
        "seed": run_args.get("seed", checkpoint.get("seed", "")),
        "epoch": checkpoint.get("epoch", run_args.get("epochs", "")),
        "checkpoint_path": str(checkpoint_path),
        "args_path": str(args.run_dir / "args.json"),
        "config_path": str(args.run_dir / "config.yaml"),
        "test_mse": result["test_mse"],
        "step_rel_l2": result["step_rel_l2"],
        "full_rollout_rel_l2": result["full_rollout_rel_l2"],
        "rollout_growth_slope": result["rollout_growth_slope"],
        "rollout_slope_fit_start": result["rollout_slope_fit_start"],
        "rollout_slope_fit_end": result["rollout_slope_fit_end"],
        "params": params,
        **complexity,
        "spectral_band_config": DEFAULT_BANDS,
        **spectral,
    }
    write_json(out_dir / "args.json", {"evaluation_args": vars(args), "source_args": run_args})
    (out_dir / "config.yaml").write_text(yaml.safe_dump(to_builtin(summary), sort_keys=True), encoding="utf-8")
    write_rollout_error(out_dir / "rollout_error_by_step.csv", result)
    with (out_dir / "complexity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["params", "peak_memory_gb", "inference_ms_per_step", "rollout_ms", "warmup", "repeats"])
        writer.writeheader()
        writer.writerow({"params": params, **complexity, "warmup": args.warmup, "repeats": args.repeats})
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
    write_json(out_dir / "evaluation_summary.json", summary)
    print(f"wrote evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
