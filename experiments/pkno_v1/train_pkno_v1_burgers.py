from __future__ import annotations
import argparse
from pathlib import Path
from common import add_common_args, run_training
from pkno.data.pkno_v1_loaders import build_burgers_v1_loaders

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train PKNO_v1 on Burgers")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--sub", type=int, default=32)
    parser.add_argument("--split-mode", choices=["tuning", "final"], default="final")
    add_common_args(parser)
    parser.set_defaults(batch_size=64, lr=1e-3, basis_kind="burgers", one_step_epochs=500, short_rollout_epochs=0)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    run_training(args, build_burgers_v1_loaders(data_path=args.data_path, batch_size=args.batch_size, sub=args.sub, split_mode=args.split_mode, num_workers=args.num_workers))

if __name__ == "__main__": main()
