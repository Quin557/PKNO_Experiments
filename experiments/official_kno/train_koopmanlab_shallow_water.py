from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import h5py
import torch
import yaml

from koopmanlab_utils import koopmanlab_optional_output_flag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train official KoopmanLab KNO on shallow-water")
    parser.add_argument("--koopmanlab-root", type=Path, default=Path("external/KoopmanLab"))
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage0_kno_baseline"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--t-in", type=int, default=10)
    parser.add_argument("--t-out", type=int, default=40)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--ntrain", type=int, default=900)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--operator-size", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--decompose", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
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


def _validate_split(args: argparse.Namespace) -> None:
    if args.ntrain <= 0 or args.ntest <= 0:
        raise ValueError("--ntrain and --ntest must both be positive.")
    if args.t_in <= 0 or args.t_out <= 0:
        raise ValueError("--t-in and --t-out must both be positive.")
    if args.sub <= 0:
        raise ValueError("--sub must be positive.")


def _sample_to_xy_time(sample: Any, required_steps: int, sub: int, key: str) -> torch.Tensor:
    if sample.ndim == 4:
        if sample.shape[-1] < 1:
            raise ValueError(f"{key} has no channel dimension: shape={sample.shape}")
        if sample.shape[0] < required_steps:
            raise ValueError(f"{key} has only {sample.shape[0]} time steps; need {required_steps}.")
        sample = sample[:required_steps, ::sub, ::sub, 0]
    elif sample.ndim == 3:
        if sample.shape[0] < required_steps:
            raise ValueError(f"{key} has only {sample.shape[0]} time steps; need {required_steps}.")
        sample = sample[:required_steps, ::sub, ::sub]
    else:
        raise ValueError(f"{key} must have shape (T, X, Y, C) or (T, X, Y); got {sample.shape}.")
    return torch.tensor(sample, dtype=torch.float32).permute(1, 2, 0)


def _load_root_data_dataset(dataset: h5py.Dataset, args: argparse.Namespace) -> torch.Tensor:
    total = args.ntrain + args.ntest
    required_steps = args.t_in + args.t_out

    if dataset.shape[0] < total:
        raise ValueError(
            f"Root /data has only {dataset.shape[0]} samples; need {total}. "
            "Use smaller --ntrain/--ntest or check the data file."
        )

    if dataset.ndim == 4 and dataset.shape[-1] >= required_steps:
        data = dataset[:total, :: args.sub, :: args.sub, :required_steps]
        return torch.tensor(data, dtype=torch.float32)

    if dataset.ndim == 4 and dataset.shape[1] >= required_steps and dataset.shape[2] == dataset.shape[3]:
        data = dataset[:total, :required_steps, :: args.sub, :: args.sub]
        return torch.tensor(data, dtype=torch.float32).permute(0, 2, 3, 1)

    if dataset.ndim == 5 and dataset.shape[1] >= required_steps and dataset.shape[-1] >= 1:
        data = dataset[:total, :required_steps, :: args.sub, :: args.sub, 0]
        return torch.tensor(data, dtype=torch.float32).permute(0, 2, 3, 1)

    raise ValueError(
        "Root /data must have shape (B, X, Y, T), (B, T, X, Y), or (B, T, X, Y, C); "
        f"got {dataset.shape}."
    )


def _load_grouped_pdebench_data(handle: h5py.File, args: argparse.Namespace) -> torch.Tensor:
    total = args.ntrain + args.ntest
    required_steps = args.t_in + args.t_out
    keys = sorted(k for k in handle.keys() if isinstance(handle[k], h5py.Group) and "data" in handle[k])

    if len(keys) < total:
        raise ValueError(
            f"Grouped shallow-water file has only {len(keys)} samples; need {total}. "
            "Use smaller --ntrain/--ntest or check that 2D_rdb_NA_NA.h5 is complete."
        )

    selected_keys = keys[:total]
    first = handle[f"{selected_keys[0]}/data"]
    first_sample = _sample_to_xy_time(
        first[:],
        required_steps=required_steps,
        sub=args.sub,
        key=f"{selected_keys[0]}/data",
    )
    data = torch.empty((total, *first_sample.shape), dtype=torch.float32)
    data[0] = first_sample

    for index, key in enumerate(selected_keys[1:], start=1):
        data[index] = _sample_to_xy_time(
            handle[f"{key}/data"][:],
            required_steps=required_steps,
            sub=args.sub,
            key=f"{key}/data",
        )
        if (index + 1) % 50 == 0 or index + 1 == total:
            print(f"loaded grouped PDEBench shallow-water samples: {index + 1}/{total}")

    print("PDEBench grouped shallow-water HDF5 loaded:", selected_keys[0], "to", selected_keys[-1])
    return data


def build_shallow_water_loaders(args: argparse.Namespace):
    _validate_split(args)

    with h5py.File(args.data_path) as handle:
        if "data" in handle and isinstance(handle["data"], h5py.Dataset):
            data = _load_root_data_dataset(handle["data"], args)
            print("Root /data shallow-water HDF5 loaded.")
        else:
            data = _load_grouped_pdebench_data(handle, args)

    train_a = data[: args.ntrain, :, :, : args.t_in]
    train_u = data[: args.ntrain, :, :, args.t_in : args.t_in + args.t_out]
    test_a = data[-args.ntest :, :, :, : args.t_in]
    test_u = data[-args.ntest :, :, :, args.t_in : args.t_in + args.t_out]

    print("Shallow Water Equations Dataset has been loaded successfully!")
    print(f"split: ntrain={args.ntrain}, ntest={args.ntest}")
    print("X train shape:", train_a.shape, "Y train shape:", train_u.shape)
    print("X test shape:", test_a.shape, "Y test shape:", test_u.shape)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u),
        batch_size=args.batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


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

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "rollout_mse_mean", "params", "source"])
        writer.writeheader()
        writer.writerow(
            {
                "epoch": args.epochs,
                "rollout_mse_mean": float(time_error.mean().item()),
                "params": model.params,
                "source": "official_koopmanlab_wrapper",
            }
        )

    torch.save({"time_error": time_error, "params": model.params}, out_dir / "time_error.pt")


if __name__ == "__main__":
    main()
