from __future__ import annotations
import argparse
from pathlib import Path
from common import add_common_args, existing_file, run_training
from pkno.data.pkno_v1_loaders import build_shallow_water_v1_loaders

def main() -> None:
    p = argparse.ArgumentParser("Train PKNO_v2-A on shallow-water"); p.add_argument("--data-path", type=existing_file, required=True); p.add_argument("--t-in", type=int, default=10); p.add_argument("--t-out", type=int, default=40); p.add_argument("--sub", type=int, default=1); p.add_argument("--dt", type=float, default=0.01); p.add_argument("--split-mode", choices=["tuning", "final"], default="final"); add_common_args(p); p.set_defaults(batch_size=5, lr=5e-5, basis_kind="shallow_water", decompose=4, delta_scale=0.005, max_grad_norm=0.1)
    a = p.parse_args(); run_training(a, build_shallow_water_v1_loaders(data_path=a.data_path, batch_size=a.batch_size, t_in=a.t_in, t_out=a.t_out, sub=a.sub, dt=a.dt, split_mode=a.split_mode, num_workers=a.num_workers))

if __name__ == "__main__": main()
