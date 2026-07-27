# Data Card: Shallow Water

## Source

- Primary alignment: KoopmanLab loader.
- KoopmanLab README says shallow-water data can be used through its data interface, and the preparation report marks PDEBench shallow-water as a later useful source.

## Stage

Stage 0/1 if data is easy to obtain; otherwise defer.

## KoopmanLab Loader

```python
kp.data.shallow_water(path, batch_size=5, T_in=10, T_out=40, sub=1)
```

## Placement

```text
$DATA_ROOT/shallow_water/
```

## Known Pitfalls

- Source URL and exact HDF5/MAT field names need confirmation before running.
