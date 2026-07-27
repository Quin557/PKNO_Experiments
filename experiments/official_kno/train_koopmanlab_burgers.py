from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train official KoopmanLab KNO on Burgers")
    parser.add_argument("--koopmanlab-root", type=Path, default=Path("external/KoopmanLab"))
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage0_kno_baseline"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sub", type=int, default=32)
    parser.add_argument("--operator-size", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--decompose", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


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
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["data_path"] = str(args.data_path)
    config["koopmanlab_root"] = str(args.koopmanlab_root)
    config["device_resolved"] = str(device)
    write_json(out_dir / "args.json", config)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    write_env(out_dir / "env.txt")

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
    model.compile()
    model.opt_init("Adam", lr=args.lr, step_size=args.step_size, gamma=args.gamma)
    model.train_single(epochs=args.epochs, trainloader=train_loader, evalloader=test_loader)
    test_mse = float(model.test_single(test_loader))

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "test_mse", "params", "source"])
        writer.writeheader()
        writer.writerow(
            {
                "epoch": args.epochs,
                "test_mse": test_mse,
                "params": model.params,
                "source": "official_koopmanlab_wrapper",
            }
        )


if __name__ == "__main__":
    main()
