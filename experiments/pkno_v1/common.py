from __future__ import annotations

import argparse
import random
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

from pkno.data.pkno_v1_loaders import V1LoaderBundle
from pkno.models.pkno_v1 import PKNOV11d, PKNOV12d
from pkno.trainers.train_pkno_v1 import PKNOV1TrainConfig, train_pkno_v1
from pkno.trainers.train_rollout import write_json


def to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage3_2_pkno_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--operator-size", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--decompose", type=int, default=8)
    parser.add_argument("--dictionary-hidden-dim", type=int, default=128)
    parser.add_argument("--dictionary-depth", type=int, default=2)
    parser.add_argument("--basis-kind", choices=["generic", "burgers", "navier_stokes", "shallow_water"], default="generic")
    parser.add_argument("--condition-embed-dim", type=int, default=128)
    parser.add_argument("--state-embed-dim", type=int, default=64)
    parser.add_argument("--koopman-hidden-dim", type=int, default=128)
    parser.add_argument("--koopman-depth", type=int, default=2)
    parser.add_argument("--delta-scale", type=float, default=0.05)
    parser.add_argument("--gate-max", type=float, default=0.1)
    parser.add_argument("--direct-prediction", action="store_true", help="Ablation: remove the physical-field residual output.")
    parser.add_argument("--pred-weight", type=float, default=5.0)
    parser.add_argument("--recon-weight", type=float, default=0.5)
    parser.add_argument("--state-weight", type=float, default=1e-4)
    parser.add_argument("--smooth-weight", type=float, default=1e-4)
    parser.add_argument("--growth-weight", type=float, default=0.0)
    parser.add_argument("--growth-quantile", type=float, default=0.99)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--one-step-epochs", type=int, default=50)
    parser.add_argument("--short-rollout-epochs", type=int, default=50)
    parser.add_argument("--short-horizon-a", type=int, default=5)
    parser.add_argument("--short-horizon-b", type=int, default=10)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--skip-test-evaluation", action="store_true", help="For validation-only tuning runs; prevents any test-set read.")
    parser.add_argument("--log-every", type=int, default=1)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def _condition_scale(bundle: V1LoaderBundle) -> torch.Tensor:
    dataset = bundle.train_loader.dataset
    if not isinstance(dataset, torch.utils.data.TensorDataset):
        return torch.ones(bundle.condition_dim)
    values = dataset.tensors[2]
    return values.abs().mean(dim=0).clamp_min(0.1)


def build_model(args: argparse.Namespace, bundle: V1LoaderBundle) -> torch.nn.Module:
    cls = PKNOV11d if bundle.spatial_dim == 1 else PKNOV12d
    model = cls(
        input_channels=bundle.input_channels, output_channels=bundle.output_channels, condition_dim=bundle.condition_dim,
        observable_dim=args.operator_size, modes=args.modes, decompose=args.decompose,
        dictionary_hidden_dim=args.dictionary_hidden_dim, dictionary_depth=args.dictionary_depth,
        basis_kind=args.basis_kind, condition_embed_dim=args.condition_embed_dim, state_embed_dim=args.state_embed_dim,
        koopman_hidden_dim=args.koopman_hidden_dim, koopman_depth=args.koopman_depth,
        delta_scale=args.delta_scale, gate_max=args.gate_max, residual_prediction=not args.direct_prediction,
    )
    model.set_condition_scale(_condition_scale(bundle))
    return model


def run_training(args: argparse.Namespace, bundle: V1LoaderBundle) -> torch.nn.Module:
    if args.modes != 16:
        raise ValueError("PKNO_v1 paper-comparison route fixes --modes 16.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    out_dir = args.output_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args, bundle)
    config: dict[str, Any] = vars(args).copy()
    config.update({"stage": "stage3_2_pkno_v1", "model": "PKNO_v1", "dataset_metadata": bundle.metadata, "input_channels": bundle.input_channels, "output_channels": bundle.output_channels, "condition_dim": bundle.condition_dim, "spatial_dim": bundle.spatial_dim, "t_out_resolved": bundle.t_out})
    write_json(out_dir / "args.json", config)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(to_builtin(config), sort_keys=True), encoding="utf-8")
    train_pkno_v1(
        model, bundle.train_loader, bundle.val_loader, bundle.test_loader,
        out_dir=out_dir, device=resolve_device(args.device), t_out=bundle.t_out,
        config=PKNOV1TrainConfig(
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, step_size=args.step_size, gamma=args.gamma,
            pred_weight=args.pred_weight, recon_weight=args.recon_weight, state_weight=args.state_weight,
            smooth_weight=args.smooth_weight, growth_weight=args.growth_weight, growth_quantile=args.growth_quantile,
            max_grad_norm=args.max_grad_norm, save_checkpoint=args.save_checkpoint, log_every=args.log_every,
            one_step_epochs=args.one_step_epochs, short_rollout_epochs=args.short_rollout_epochs,
            short_horizons=(args.short_horizon_a, args.short_horizon_b), resume=args.resume,
            evaluate_test=not args.skip_test_evaluation,
        ),
    )
    return model
