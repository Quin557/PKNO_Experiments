from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ikno.models import IKNO1d, IKNO2d
from pkno.data.stage3_loaders import LoaderBundle
from pkno.trainers.train_rollout import RolloutTrainConfig, train_autoregressive, write_json


def to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ikno_baseline"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--operator-size", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument(
        "--decompose",
        type=int,
        default=4,
        help="Number of IKNO neural-operator iterations (paper default l=4).",
    )
    parser.add_argument("--koopman-power", type=int, default=2)
    parser.add_argument("--inn-blocks", type=int, default=4)
    parser.add_argument("--inn-hidden-dim", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def build_model(args: argparse.Namespace, bundle: LoaderBundle) -> torch.nn.Module:
    model_cls = IKNO1d if bundle.spatial_dim == 1 else IKNO2d
    return model_cls(
        input_channels=bundle.input_channels,
        output_channels=bundle.output_channels,
        observable_dim=args.operator_size,
        modes=args.modes,
        operator_layers=args.decompose,
        koopman_power=args.koopman_power,
        inn_blocks=args.inn_blocks,
        inn_hidden_dim=args.inn_hidden_dim,
    )


def run_ikno_training(args: argparse.Namespace, bundle: LoaderBundle) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.output_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, bundle)
    config: dict[str, Any] = vars(args).copy()
    config.update(
        {
            "stage": "ikno_baseline",
            "model": "IKNO",
            "dataset_metadata": bundle.metadata,
            "input_channels": bundle.input_channels,
            "output_channels": bundle.output_channels,
            "condition_dim": bundle.condition_dim,
            "spatial_dim": bundle.spatial_dim,
            "t_out_resolved": bundle.t_out,
            "dictionary": "pointwise_invertible_residual_coupling",
            "prediction_weight": 5.0,
            "reconstruction_weight": 0.0,
            "reconstruction_loss": False,
        }
    )
    write_json(out_dir / "args.json", config)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(to_builtin(config), sort_keys=True), encoding="utf-8")

    train_config = RolloutTrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        recon_weight=0.0,
        max_grad_norm=args.max_grad_norm,
        save_checkpoint=args.save_checkpoint,
        log_every=args.log_every,
    )
    train_autoregressive(
        model,
        bundle.train_loader,
        bundle.test_loader,
        out_dir=out_dir,
        device=resolve_device(args.device),
        config=train_config,
        t_out=bundle.t_out,
        output_channels=bundle.output_channels,
    )
