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

from pkno.data.stage3_loaders import LoaderBundle
from pkno.trainers.train_rollout import RolloutTrainConfig, train_autoregressive, write_json
from pkno_u.diagnostics import collect_rollout_diagnostics, write_rollout_diagnostics
from pkno_u.models import PKNOU1d, PKNOU2d


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage3_1_param_kno_u"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
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
    parser.add_argument(
        "--basis-kind",
        type=str,
        default="generic",
        choices=["generic", "burgers", "navier_stokes", "shallow_water"],
    )
    parser.add_argument("--condition-mode", choices=["physical_only", "physical_compact_state", "physical_gated_state"], default="physical_only")
    parser.add_argument("--condition-embed-dim", type=int, default=128)
    parser.add_argument("--state-embed-dim", type=int, default=16)
    parser.add_argument("--koopman-hidden-dim", type=int, default=128)
    parser.add_argument("--koopman-depth", type=int, default=2)
    parser.add_argument("--delta-scale", type=float, default=0.05)
    parser.add_argument("--max-operator-norm", type=float, default=0.98)
    parser.add_argument("--unet-start-layer", type=int, default=None)
    parser.add_argument("--unet-base-channels", type=int, default=32)
    parser.add_argument("--hf-cutoff", type=float, default=0.5)
    parser.add_argument("--hf-residual-scale", type=float, default=0.05)
    parser.add_argument("--no-checkpoint-unet", dest="checkpoint_unet", action="store_false")
    parser.set_defaults(checkpoint_unet=True)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--diagnostic-batches", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1)


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def _build_model(args: argparse.Namespace, bundle: LoaderBundle) -> torch.nn.Module:
    model_cls = PKNOU1d if bundle.spatial_dim == 1 else PKNOU2d
    return model_cls(
        input_channels=bundle.input_channels,
        output_channels=bundle.output_channels,
        condition_dim=bundle.condition_dim,
        observable_dim=args.operator_size,
        modes=args.modes,
        decompose=args.decompose,
        dictionary_hidden_dim=args.dictionary_hidden_dim,
        dictionary_depth=args.dictionary_depth,
        basis_kind=args.basis_kind,
        condition_embed_dim=args.condition_embed_dim,
        state_embed_dim=args.state_embed_dim,
        koopman_hidden_dim=args.koopman_hidden_dim,
        koopman_depth=args.koopman_depth,
        delta_scale=args.delta_scale,
        max_operator_norm=args.max_operator_norm,
        condition_mode=args.condition_mode,
        unet_start_layer=args.unet_start_layer,
        unet_base_channels=args.unet_base_channels,
        hf_cutoff=args.hf_cutoff,
        hf_residual_scale=args.hf_residual_scale,
        checkpoint_unet=args.checkpoint_unet,
    )


def run_stage3_1_training(args: argparse.Namespace, bundle: LoaderBundle) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    out_dir = args.output_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = _build_model(args, bundle)
    config: dict[str, Any] = vars(args).copy()
    config.update(
        {
            "stage": "stage3_1_param_kno_u",
            "model": "PKNO-U",
            "training_loss": "5 * rollout_prediction_mse + 0.5 * reconstruction_mse",
            "evaluation_primary_metric": "full_rollout_relative_l2",
            "dataset_metadata": bundle.metadata,
            "input_channels": bundle.input_channels,
            "output_channels": bundle.output_channels,
            "condition_dim": bundle.condition_dim,
            "spatial_dim": bundle.spatial_dim,
            "t_out_resolved": bundle.t_out,
        }
    )
    write_json(out_dir / "args.json", config)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(_to_builtin(config), sort_keys=True), encoding="utf-8")
    train_config = RolloutTrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        max_grad_norm=args.max_grad_norm,
        save_checkpoint=args.save_checkpoint,
        log_every=args.log_every,
    )
    train_autoregressive(
        model,
        bundle.train_loader,
        bundle.test_loader,
        out_dir=out_dir,
        device=device,
        config=train_config,
        t_out=bundle.t_out,
        output_channels=bundle.output_channels,
    )
    diagnostics = collect_rollout_diagnostics(
        model,
        bundle.test_loader,
        device=device,
        t_out=bundle.t_out,
        max_batches=args.diagnostic_batches,
    )
    write_rollout_diagnostics(out_dir / "stability_diagnostics.csv", diagnostics)
