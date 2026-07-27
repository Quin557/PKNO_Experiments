# Code Map: KoopmanLab

## Source

- Repository: https://github.com/Koopman-Laboratory/KoopmanLab
- Local path: `external/KoopmanLab`
- Current local commit: `c9e347a9df50103d308235148132ce7ad1c850a4`
- License: GPL-3.0 according to the repository license file.

## Stage 0 Role

Primary source for KNO baseline reproduction. Use this before local reimplementation.

## Key Files

| File | Purpose |
|---|---|
| `demo_ns.py` | Official Navier-Stokes workflow example. |
| `koopmanlab/models/kno.py` | Compact KNO model implementation. |
| `koopmanlab/model.py` | High-level train/test wrapper and loss composition. |
| `koopmanlab/data.py` | Burgers, shallow-water, Navier-Stokes data interfaces. |
| `README.md` | Official install, data links, and usage notes. |

## Official Data Interfaces

From the README:

```python
kp.data.burgers(path, batch_size=64, sub=32)
kp.data.shallow_water(path, batch_size=5, T_in=10, T_out=40, sub=1)
kp.data.navier_stokes(path, batch_size=10, T_in=10, T_out=40, type="1e-3", sub=1)
```

## Baseline Policy

Run official KoopmanLab commands first. If the official script lacks structured metric outputs, wrap or post-process it instead of changing the model immediately.
