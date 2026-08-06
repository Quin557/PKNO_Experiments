from __future__ import annotations
import argparse
from pathlib import Path
from common import add_common_args, run_training
from pkno.data.pkno_v1_loaders import build_burgers_v1_loaders

def main() -> None:
    p = argparse.ArgumentParser("Train PKNO_v2-A on Burgers"); p.add_argument("--data-path", type=Path, required=True); p.add_argument("--sub", type=int, default=32); p.add_argument("--split-mode", choices=["tuning", "final"], default="final"); add_common_args(p); p.set_defaults(batch_size=64, lr=1e-3, basis_kind="burgers", decompose=8)
    a = p.parse_args(); run_training(a, build_burgers_v1_loaders(data_path=a.data_path, batch_size=a.batch_size, sub=a.sub, split_mode=a.split_mode, num_workers=a.num_workers))

if __name__ == "__main__": main()
