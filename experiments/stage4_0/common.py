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

from ampkno.models import AMPKNO1d, AMPKNO2d
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage4_0_am_pkno"))
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
    parser.add_argument("--decompose", type=int, default=8)
    parser.add_argument(
        "--max-modes",
        type=int,
        default=0,
        help=(
            "Compute cap for generated Fourier modes. 0 uses every available FFT mode; "
            "this is not KNO's per-mode parameter table size."
        ),
    )
    parser.add_argument("--frequency-basis-dim", type=int, default=32)
    parser.add_argument("--dictionary-hidden-dim", type=int, default=128)
    parser.add_argument("--dictionary-depth", type=int, default=2)
    parser.add_argument(
        "--basis-kind",
        type=str,
        default="generic",
        choices=["generic", "burgers", "navier_stokes", "shallow_water"],
        help="Condition-independent fixed observables prepended to learned dictionary channels.",
    )
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--condition-embed-dim", type=int, default=128)
    parser.add_argument("--state-embed-dim", type=int, default=64)
    parser.add_argument("--generator-hidden-dim", type=int, default=128)
    parser.add_argument("--generator-depth", type=int, default=2)
    parser.add_argument("--output-scale", type=float, default=0.02)
    parser.add_argument(
        "--operator-factorization",
        type=str,
        default="factorized",
        choices=["factorized", "full"],
        help="2D AM-PKNO generator type. 1D always uses the direct conditioned generator.",
    )
    parser.add_argument("--factorized-rank", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--use-hf-residual", action="store_true")
    parser.add_argument("--hf-hidden-dim", type=int, default=32)
    parser.add_argument("--hf-residual-scale", type=float, default=0.1)
    parser.add_argument(
        "--no-checkpoint-koopman",
        dest="checkpoint_koopman",
        action="store_false",
        help="Disable activation checkpointing around repeated Koopman updates. Faster but uses more memory.",
    )
    parser.set_defaults(checkpoint_koopman=True)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def build_model(args: argparse.Namespace, bundle: LoaderBundle) -> torch.nn.Module:
    model_cls = AMPKNO1d if bundle.spatial_dim == 1 else AMPKNO2d
    return model_cls(
        input_channels=bundle.input_channels,
        output_channels=bundle.output_channels,
        condition_dim=bundle.condition_dim,
        observable_dim=args.operator_size,
        decompose=args.decompose,
        max_modes=args.max_modes,
        frequency_basis_dim=args.frequency_basis_dim,
        dictionary_hidden_dim=args.dictionary_hidden_dim,
        dictionary_depth=args.dictionary_depth,
        basis_kind=args.basis_kind,
        decoder_hidden_dim=args.decoder_hidden_dim,
        condition_embed_dim=args.condition_embed_dim,
        state_embed_dim=args.state_embed_dim,
        generator_hidden_dim=args.generator_hidden_dim,
        generator_depth=args.generator_depth,
        output_scale=args.output_scale,
        operator_factorization=args.operator_factorization,
        factorized_rank=args.factorized_rank,
        use_hf_residual=args.use_hf_residual,
        hf_hidden_dim=args.hf_hidden_dim,
        hf_residual_scale=args.hf_residual_scale,
        checkpoint_koopman=args.checkpoint_koopman,
    )


def run_stage4_training(args: argparse.Namespace, bundle: LoaderBundle) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.output_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, bundle)
    config: dict[str, Any] = vars(args).copy()
    config["stage"] = "stage4_0_am_pkno"
    config["model"] = "AM-PKNO"
    config["dataset_metadata"] = bundle.metadata
    config["input_channels"] = bundle.input_channels
    config["output_channels"] = bundle.output_channels
    config["condition_dim"] = bundle.condition_dim
    config["spatial_dim"] = bundle.spatial_dim
    config["t_out_resolved"] = bundle.t_out
    config["mode_policy"] = "all_fft_modes" if args.max_modes <= 0 else "compute_capped"
    config["stage4_0_rule"] = (
        "shared_dictionary_conditioned_am_koopman"
        if bundle.spatial_dim == 1
        else f"shared_dictionary_conditioned_am_koopman_{args.operator_factorization}"
    )
    write_json(out_dir / "args.json", config)
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(to_builtin(config), sort_keys=True),
        encoding="utf-8",
    )

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
        device=resolve_device(args.device),
        config=train_config,
        t_out=bundle.t_out,
        output_channels=bundle.output_channels,
    )
