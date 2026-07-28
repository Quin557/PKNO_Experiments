from __future__ import annotations

import argparse
from pathlib import Path

from common import add_common_args, run_stage3_training

from pkno.data.stage3_loaders import build_burgers_stage3_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train Stage3_0 Param-KNO on Burgers")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--sub", type=int, default=32)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    add_common_args(parser)
    parser.set_defaults(batch_size=64, lr=1e-3, epochs=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_burgers_stage3_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        sub=args.sub,
        ntrain=args.ntrain,
        ntest=args.ntest,
        num_workers=args.num_workers,
    )
    run_stage3_training(args, bundle)


if __name__ == "__main__":
    main()
