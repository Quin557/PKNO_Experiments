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
from pkno_v2.model import PKNOV2_1d, PKNOV2_2d
from pkno_v2.trainer import V2TrainConfig, train_v2
from pkno.trainers.train_rollout import write_json


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage3_3_pkno_v2"))
    parser.add_argument("--device", default="cuda"); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500); parser.add_argument("--batch-size", type=int, default=None); parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=None, help="Protocol assertion; loaders keep fixed historical indices.")
    parser.add_argument("--ntest", type=int, default=None, help="Protocol assertion; loaders keep fixed historical indices.")
    parser.add_argument("--lr", type=float, default=5e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=100); parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--operator-size", type=int, default=32); parser.add_argument("--modes", type=int, default=16); parser.add_argument("--decompose", type=int, default=8)
    parser.add_argument("--dictionary-hidden-dim", type=int, default=128); parser.add_argument("--dictionary-depth", type=int, default=2)
    parser.add_argument("--basis-kind", choices=["generic", "burgers", "navier_stokes", "shallow_water"], default="generic")
    parser.add_argument("--condition-embed-dim", type=int, default=64); parser.add_argument("--state-embed-dim", type=int, default=16)
    parser.add_argument("--koopman-hidden-dim", type=int, default=128); parser.add_argument("--koopman-depth", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4); parser.add_argument("--delta-scale", type=float, default=0.02); parser.add_argument("--eta-max", type=float, default=0.5)
    parser.add_argument("--hf-hidden", type=int, default=16)
    parser.add_argument("--recon-weight", type=float, default=0.5); parser.add_argument("--gradient-weight", type=float, default=1e-3); parser.add_argument("--spectral-weight", type=float, default=1e-4); parser.add_argument("--gate-weight", type=float, default=1e-4); parser.add_argument("--late-final-weight", type=float, default=2.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0); parser.add_argument("--save-checkpoint", action="store_true"); parser.add_argument("--log-every", type=int, default=1)


def resolve_device(name: str) -> torch.device:
    return torch.device("cpu" if name == "cuda" and not torch.cuda.is_available() else name)


def _condition_scale(bundle: V1LoaderBundle) -> torch.Tensor:
    dataset = bundle.train_loader.dataset
    if isinstance(dataset, torch.utils.data.TensorDataset):
        return dataset.tensors[2].abs().mean(dim=0).clamp_min(0.1)
    return torch.ones(bundle.condition_dim)


def _builtin(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {k: _builtin(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_builtin(v) for v in value]
    return value


def run_training(args: argparse.Namespace, bundle: V1LoaderBundle) -> None:
    if args.modes != 16:
        raise ValueError("PKNO_v2 paper-comparison route fixes --modes 16.")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    out_dir = args.output_dir / args.run_name; out_dir.mkdir(parents=True, exist_ok=True)
    if args.ntrain is not None or args.ntest is not None:
        expected_train = 900 if bundle.spatial_dim == 2 and bundle.metadata.get("dataset") == "shallow_water" else 1000
        expected_test = 100 if bundle.spatial_dim == 2 and bundle.metadata.get("dataset") == "shallow_water" else 200
        if args.ntrain not in (None, expected_train) or args.ntest not in (None, expected_test):
            raise ValueError(f"PKNO_v2 fixed protocol is ntrain/ntest={expected_train}/{expected_test}")
    cls = PKNOV2_1d if bundle.spatial_dim == 1 else PKNOV2_2d
    model = cls(input_channels=bundle.input_channels, output_channels=bundle.output_channels, condition_dim=bundle.condition_dim,
                observable_dim=args.operator_size, modes=args.modes, decompose=args.decompose, dictionary_hidden_dim=args.dictionary_hidden_dim,
                dictionary_depth=args.dictionary_depth, basis_kind=args.basis_kind, condition_embed_dim=args.condition_embed_dim,
                state_embed_dim=args.state_embed_dim, koopman_hidden_dim=args.koopman_hidden_dim, koopman_depth=args.koopman_depth,
                rank=args.rank, delta_scale=args.delta_scale, eta_max=args.eta_max, hf_hidden=args.hf_hidden)
    model.set_condition_scale(_condition_scale(bundle))
    config: dict[str, Any] = _builtin(vars(args).copy()); config.update({"stage": "stage3_3_pkno_v2", "model": "PKNO_v2-A", "dataset_metadata": bundle.metadata, "input_channels": bundle.input_channels, "output_channels": bundle.output_channels, "condition_dim": bundle.condition_dim, "spatial_dim": bundle.spatial_dim, "t_out_resolved": bundle.t_out})
    write_json(out_dir / "args.json", config); (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    train_v2(model, bundle.train_loader, bundle.val_loader, bundle.test_loader, out_dir=out_dir, device=resolve_device(args.device), config=V2TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, step_size=args.step_size, gamma=args.gamma, recon_weight=args.recon_weight, gradient_weight=args.gradient_weight, spectral_weight=args.spectral_weight, gate_weight=args.gate_weight, late_final_weight=args.late_final_weight, max_grad_norm=args.max_grad_norm, save_checkpoint=args.save_checkpoint, log_every=args.log_every), t_out=bundle.t_out)
