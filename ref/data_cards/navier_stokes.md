# Data Card: Navier-Stokes

## Source

- Primary alignment: KoopmanLab `demo_ns.py` and `kp.data.navier_stokes`.
- KoopmanLab README provides a Google Drive dataset folder.

## Stage

Stage 0.

## Purpose

- Main official KNO baseline.
- Long rollout.
- High-frequency and vorticity/spectrum analysis.

## Placement

```text
$DATA_ROOT/navier_stokes/
```

## KoopmanLab Loader

```python
kp.data.navier_stokes(path, batch_size=10, T_in=10, T_out=40, type="1e-3", sub=1)
```

## Shape

KoopmanLab converts data to:

```text
[B, X, Y, T]
```

## Known Pitfalls

- Viscosity type matters: `1e-3`, `1e-4`, `1e-5`.
- Do not compare different viscosity files as one benchmark.
- Record the exact official data file and viscosity setting.
