from __future__ import annotations
import argparse
from pathlib import Path
from common import add_common_args, run_training
from pkno.data.pkno_v1_loaders import build_ns_v1_loaders

def main() -> None:
    p = argparse.ArgumentParser("Train PKNO_v2-A on NS v1e-4"); p.add_argument("--data-path", type=Path, required=True); p.add_argument("--t-in", type=int, default=10); p.add_argument("--t-out", type=int, default=40); p.add_argument("--sub", type=int, default=1); p.add_argument("--dt", type=float, default=1.0); add_common_args(p); p.set_defaults(batch_size=10, lr=5e-4, basis_kind="navier_stokes", decompose=8, gate_weight=2e-4)
    a = p.parse_args(); run_training(a, build_ns_v1_loaders(data_path=a.data_path, viscosity_type="1e-4", batch_size=a.batch_size, t_in=a.t_in, t_out=a.t_out, sub=a.sub, dt=a.dt, num_workers=a.num_workers))

if __name__ == "__main__": main()
