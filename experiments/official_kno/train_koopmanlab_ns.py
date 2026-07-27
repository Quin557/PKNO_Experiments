from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from koopmanlab_utils import koopmanlab_optional_output_flag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train official KoopmanLab KNO on Navier-Stokes")
    parser.add_argument("--koopmanlab-root", type=Path, default=Path("external/KoopmanLab"))
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--t-in", type=int, default=10)
    parser.add_argument("--t-out", type=int, default=40)
    parser.add_argument("--viscosity-type", choices=["1e-3", "1e-4", "1e-5"], default="1e-3")
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--operator-size", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--decompose", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-plots", action="store_true")
    return parser.parse_args()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(to_builtin(obj), indent=2, sort_keys=True), encoding="utf-8")


def to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def write_env(path: Path) -> None:
    lines = [
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_version={torch.version.cuda}",
        f"cuda_device_count={torch.cuda.device_count()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"cuda_device_name={torch.cuda.get_device_name(0)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if not args.koopmanlab_root.exists():
        raise FileNotFoundError(f"KoopmanLab root not found: {args.koopmanlab_root}")
    if not args.data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    sys.path.insert(0, str(args.koopmanlab_root.resolve()))
    import koopmanlab as kp  # noqa: PLC0415

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir / args.run_name
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_plots:
        fig_dir.mkdir(parents=True, exist_ok=True)

    config = to_builtin(vars(args).copy())
    config["device_resolved"] = str(device)
    write_json(out_dir / "args.json", config)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    write_env(out_dir / "env.txt")

    train_loader, test_loader = kp.data.navier_stokes(
        str(args.data_path),
        batch_size=args.batch_size,
        T_in=args.t_in,
        T_out=args.t_out,
        type=args.viscosity_type,
        sub=args.sub,
    )

    model = kp.model.koopman(
        backbone="KNO2d",
        autoencoder="MLP",
        o=args.operator_size,
        m=args.modes,
        r=args.decompose,
        t_in=args.t_in,
        device=device,
    )
    model.compile()
    model.opt_init("Adam", lr=args.lr, step_size=args.step_size, gamma=args.gamma)
    model.train(epochs=args.epochs, trainloader=train_loader, evalloader=test_loader, T_out=args.t_out)

    time_error = model.test(
        test_loader,
        T_out=args.t_out,
        path=str(fig_dir) + "/",
        is_save=koopmanlab_optional_output_flag(args.save_plots),
        is_plot=koopmanlab_optional_output_flag(args.save_plots),
    )

    with (out_dir / "rollout_error_by_step.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "mse"])
        writer.writeheader()
        for step, value in enumerate(time_error.reshape(-1).detach().cpu().tolist()):
            writer.writerow({"step": step, "mse": value})

    final_mse = float(time_error.mean().item())
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "rollout_mse_mean", "params", "source"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": args.epochs,
                "rollout_mse_mean": final_mse,
                "params": model.params,
                "source": "official_koopmanlab_wrapper",
            }
        )

    torch.save(
        {"time_error": time_error, "params": model.params},
        out_dir / "time_error.pt",
    )


if __name__ == "__main__":
    main()
