# Data Card: Burgers

## Source

- Primary alignment: KoopmanLab data interface and official KNO data path.
- KoopmanLab loader: `kp.data.burgers(path, batch_size=64, sub=32)`.

## Stage

Stage 0.

## Purpose

- Fast KNO baseline.
- Mesh-independence check.
- Spectral metric validation.
- Early ablation testing.

## Placement

```text
$DATA_ROOT/burgers/
```

## Shape

To confirm after data download. KoopmanLab `data.py` expects fields:

```text
a
u
```

## Known Pitfalls

- Confirm exact file source and split before comparing to paper numbers.
- Record `sub` downsampling factor in every config.
