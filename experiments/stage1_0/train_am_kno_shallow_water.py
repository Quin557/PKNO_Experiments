from __future__ import annotations

import argparse
from pathlib import Path

from common import add_common_args, run_stage1_training

from pkno.data.stage3_loaders import build_shallow_water_stage3_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train Stage1_0 AM-KNO on shallow-water")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--t-in", type=int, default=10)
    parser.add_argument("--t-out", type=int, default=40)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--ntrain", type=int, default=900)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.01)
    add_common_args(parser)
    parser.set_defaults(
        batch_size=5,
        lr=2e-4,
        max_grad_norm=0.5,
        epochs=1,
        condition_mode="state",
        frequency_basis_dim=32,
        output_scale=0.015,
        max_modes=0,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_shallow_water_stage3_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        t_in=args.t_in,
        t_out=args.t_out,
        sub=args.sub,
        ntrain=args.ntrain,
        ntest=args.ntest,
        dt=args.dt,
        num_workers=args.num_workers,
    )
    run_stage1_training(args, bundle)


if __name__ == "__main__":
    main()
