# Code Map: Parametric Koopman Reference

## Source

- Repository: https://github.com/GUOYUE-Cynthia/Learning-Parametric-Koopman-Decompositions
- Local path: `external/pknn_reference`
- Current local commit: `617b8753c4f099a2faeb9a18e0ba756a08e4262b`

## Stage Role

Stage 3 reference for shared dictionaries and parameter-conditioned Koopman families.

## Key Files / Folders

| Path | Purpose |
|---|---|
| `src/koopmanlib/dictionary.py` | Dictionary / observable functions. |
| `src/koopmanlib/param_solver.py` | Parameterized Koopman solver logic. |
| `src/koopmanlib/K_structure.py` | Koopman matrix structure utilities. |
| `examples_torch/` | Torch examples, useful for porting checks. |
| `examples/ParametricKoopman/` | Toy systems and generated examples. |

## Porting Policy

Use this repository as algorithmic reference. Do not mix it directly into the KNO/PKNO PyTorch training loop until module boundaries and tests are defined.
