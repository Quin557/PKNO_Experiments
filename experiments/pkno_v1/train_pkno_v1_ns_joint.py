from __future__ import annotations
import argparse
from pathlib import Path
from common import add_common_args, run_training
from pkno.data.pkno_v1_loaders import build_joint_ns_v1_loaders
from pkno.trainers.train_pkno_v1 import evaluate_v1
from pkno.trainers.train_rollout import write_json
from torch.utils.data import DataLoader, Subset, TensorDataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train PKNO_v1 jointly on NS v1e-3 and v1e-4")
    parser.add_argument("--data-v1e3", type=Path, required=True); parser.add_argument("--data-v1e4", type=Path, required=True)
    parser.add_argument("--t-in", type=int, default=10); parser.add_argument("--t-out", type=int, default=40)
    parser.add_argument("--sub", type=int, default=1); parser.add_argument("--dt", type=float, default=1.0)
    add_common_args(parser)
    parser.set_defaults(batch_size=10, lr=5e-4, max_grad_norm=1.0, basis_kind="navier_stokes", growth_weight=1e-3)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    bundle = build_joint_ns_v1_loaders(data_v1e3=args.data_v1e3, data_v1e4=args.data_v1e4, batch_size=args.batch_size, t_in=args.t_in, t_out=args.t_out, sub=args.sub, dt=args.dt, num_workers=args.num_workers, seed=args.seed)
    model = run_training(args, bundle)
    dataset = bundle.test_loader.dataset
    if not isinstance(dataset, TensorDataset):
        raise TypeError("Joint NS test dataset must be a TensorDataset.")
    device = next(model.parameters()).device
    per_condition = {}
    for name, start in (("v1e3", 0), ("v1e4", 200)):
        loader = DataLoader(Subset(dataset, range(start, start + 200)), batch_size=args.batch_size, shuffle=False)
        stats = evaluate_v1(model, loader, device=device, t_out=bundle.t_out)
        per_condition[name] = {key: stats[key] for key in ("full_rel_l2", "step_rel_l2", "pred_mse")}
    macro = sum(item["full_rel_l2"] for item in per_condition.values()) / 2
    write_json(args.output_dir / args.run_name / "joint_test_by_condition.json", {"per_condition": per_condition, "macro_full_rel_l2": macro})

if __name__ == "__main__": main()
