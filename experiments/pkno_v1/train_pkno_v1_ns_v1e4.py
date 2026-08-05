from __future__ import annotations
import argparse
from pathlib import Path
from common import add_common_args, run_training
from pkno.data.pkno_v1_loaders import build_ns_v1_loaders

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train PKNO_v1 on NS v1e-4")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--t-in", type=int, default=10); parser.add_argument("--t-out", type=int, default=40)
    parser.add_argument("--sub", type=int, default=1); parser.add_argument("--dt", type=float, default=1.0)
    add_common_args(parser)
    parser.set_defaults(batch_size=10, lr=5e-4, max_grad_norm=1.0, basis_kind="navier_stokes", growth_weight=1e-3)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    run_training(args, build_ns_v1_loaders(data_path=args.data_path, viscosity_type="1e-4", batch_size=args.batch_size, t_in=args.t_in, t_out=args.t_out, sub=args.sub, dt=args.dt, num_workers=args.num_workers))

if __name__ == "__main__": main()
