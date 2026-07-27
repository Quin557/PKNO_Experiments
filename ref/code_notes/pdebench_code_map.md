# Code Map: PDEBench

## Source

- Repository: https://github.com/pdebench/PDEBench
- Dataset DOI: https://doi.org/10.18419/darus-2986
- Local path: `external/PDEBench`
- Current local commit: `4ff3e3a4aa1561721b5571fa3a048a0a463e0568`

## Stage Role

Later Stage 3/4 parameterized PDE data and cross-condition experiments.

## Key Files / Folders

| Path | Purpose |
|---|---|
| `pdebench/data_download/` | Dataset download scripts. |
| `pdebench/data_gen/` | Data generation scripts. |
| `pdebench/models/` | Baseline training/evaluation code. |
| `pdebench/models/metrics.py` | Reference metric implementation. |

## Notes

PDEBench is not a blocker for Stage 0 KNO baseline reproduction. Add it once Burgers/Navier-Stokes baselines and spectral metrics are stable.
