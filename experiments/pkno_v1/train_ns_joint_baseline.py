"""Joint two-viscosity NS baselines used to test physical parameterization.

This script is deliberately separate from the historical single-condition
baselines.  It trains each non-V1 model on the same balanced joint loader and
uses the joint validation set during training; the held-out test set is only
evaluated once after the final epoch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pkno.data.pkno_v1_loaders import build_joint_ns_v1_loaders
from pkno.models.param_kno import ParamKNO2d
from pkno.trainers.train_rollout import RolloutTrainConfig, evaluate_rollout, train_autoregressive, write_json, write_rollout_error, write_spectral_metrics


class LastChannelAdapter(nn.Module):
    """Adapt official KNO's T_in-channel decoder to the shared rollout API."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, history: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del condition
        prediction, reconstruction = self.model(history)
        return prediction[..., -1:], reconstruction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train non-V1 models jointly on NS v1e-3 and v1e-4")
    parser.add_argument("--model", choices=["kno", "ikno", "amkno", "pkno"], required=True)
    parser.add_argument("--data-v1e3", type=Path, required=True)
    parser.add_argument("--data-v1e4", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage3_2_pkno_v1/joint_baselines"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--t-in", type=int, default=10)
    parser.add_argument("--t-out", type=int, default=40)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--operator-size", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--decompose", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def build_model(args: argparse.Namespace, condition_dim: int) -> nn.Module:
    if args.model == "kno":
        koopman_root = ROOT / "external" / "KoopmanLab"
        if str(koopman_root) not in sys.path:
            sys.path.insert(0, str(koopman_root))
        from koopmanlab.models import KNO2d, decoder_mlp, encoder_mlp  # noqa: PLC0415
        return LastChannelAdapter(KNO2d(encoder_mlp(args.t_in, args.operator_size), decoder_mlp(args.t_in, args.operator_size), args.operator_size, modes_x=args.modes, modes_y=args.modes, decompose=args.decompose))
    if args.model == "ikno":
        from ikno.models import IKNO2d  # noqa: PLC0415
        return IKNO2d(input_channels=args.t_in, output_channels=1, observable_dim=args.operator_size, modes=args.modes, operator_layers=4, koopman_power=2, inn_blocks=4, inn_hidden_dim=128)
    if args.model == "amkno":
        from amkno.models import AMKNO2d  # noqa: PLC0415
        return AMKNO2d(input_channels=args.t_in, output_channels=1, condition_dim=condition_dim, observable_dim=args.operator_size, decompose=args.decompose, max_modes=0, condition_mode="freq", output_scale=0.03, operator_factorization="factorized", factorized_rank=1)
    return ParamKNO2d(input_channels=args.t_in, output_channels=1, condition_dim=condition_dim, observable_dim=args.operator_size, modes=args.modes, decompose=args.decompose, dictionary_hidden_dim=128, dictionary_depth=2, basis_kind="navier_stokes", condition_embed_dim=128, state_embed_dim=64, koopman_hidden_dim=128, koopman_depth=2, delta_scale=0.05)


def main() -> None:
    args = parse_args()
    if args.modes != 16:
        raise ValueError("Joint comparison fixes modes=16 for KNO-family models.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    bundle = build_joint_ns_v1_loaders(data_v1e3=args.data_v1e3, data_v1e4=args.data_v1e4, batch_size=args.batch_size, t_in=args.t_in, t_out=args.t_out, sub=args.sub, dt=args.dt, num_workers=args.num_workers, seed=args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args, bundle.condition_dim)
    config: dict[str, Any] = vars(args).copy()
    config.update({"stage": "stage3_2_joint_ns", "dataset_metadata": bundle.metadata, "validation_protocol": "val only during training; final test only after epoch 500"})
    write_json(out_dir / "args.json", config)
    (out_dir / "config.yaml").write_text(yaml.safe_dump({key: str(value) if isinstance(value, Path) else value for key, value in config.items()}, sort_keys=True), encoding="utf-8")
    train_autoregressive(model, bundle.train_loader, bundle.val_loader, out_dir=out_dir, device=device, config=RolloutTrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, step_size=args.step_size, gamma=args.gamma, max_grad_norm=args.max_grad_norm, save_checkpoint=False, log_every=args.log_every), t_out=bundle.t_out, output_channels=1)
    final = evaluate_rollout(model, bundle.test_loader, device=device, t_out=bundle.t_out, output_channels=1)
    write_rollout_error(out_dir / "test_rollout_error_by_step.csv", final["step_mse"])
    if final["first_pred"] is not None:
        write_spectral_metrics(out_dir / "test_spectral_metrics.csv", final["first_pred"], final["first_target"])
    dataset = bundle.test_loader.dataset
    if not isinstance(dataset, TensorDataset):
        raise TypeError("Joint NS test dataset must be a TensorDataset.")
    per_condition = {}
    for name, start in (("v1e3", 0), ("v1e4", 200)):
        stats = evaluate_rollout(model, DataLoader(Subset(dataset, range(start, start + 200)), batch_size=args.batch_size, shuffle=False), device=device, t_out=bundle.t_out, output_channels=1)
        per_condition[name] = {key: stats[key] for key in ("full_rel_l2", "step_rel_l2", "pred_mse")}
    macro = sum(item["full_rel_l2"] for item in per_condition.values()) / 2
    write_json(out_dir / "test_evaluation_summary.json", {"full_rel_l2": final["full_rel_l2"], "step_rel_l2": final["step_rel_l2"], "pred_mse": final["pred_mse"], "per_condition": per_condition, "macro_full_rel_l2": macro})


if __name__ == "__main__":
    main()
