# Stage3_2 PKNO_v1 Experiment Design

## Fixed Comparison Protocol

`modes=16` is fixed. Do not increase Fourier modes to obtain a PKNO_v1 gain.
Use `O=32`, Adam, `weight_decay=1e-4`, `step_size=100`, `gamma=0.5`, and seed
42 for the primary comparison.

| Task | Train | Validation | Test | Decompose |
|---|---|---|---|---:|
| Burgers | `0:999` final / `0:899` tuning | `900:999` tuning only | `1000:1199` | 8 |
| NS `1e-3` | `0:999` | `1200:1399` | `1000:1199` | 8 |
| NS `1e-4` | `0:999` | `1200:1399` | `1000:1199` | 8 |
| Shallow-water | `0:899` final / `0:799` tuning | `800:899` tuning only | `900:999` | 4 |

For Burgers and shallow-water, use the tuning split only to select settings.
Then retrain from scratch on the historical training count and evaluate the
historical test set once. NS has unused trajectories for validation, so the
historical `1000/200` train/test protocol remains unchanged.

## Promotion Gate

At seed 42, the final checkpoint must beat Stage3_0 PKNO on every primary
task before any paper replacement is considered.

| Task | Must beat old PKNO | Stretch target |
|---|---:|---:|
| Burgers T1 | `5.163e-3` | `< 5.163e-3` |
| NS `1e-3` T40 | `1.632e-2` | `< 1.061e-2` (iKNO) |
| NS `1e-4` T40 | `4.641e-1` | `< 4.641e-1` |
| Shallow-water T40 | `1.486e-2` | `< 4.725e-3` (iKNO) |

## Required Runs

1. Run smoke tests for all four tasks.
2. Run the four main PKNO_v1 configurations for 500 epochs.
3. Run targeted ablations:
   - Burgers direct output: remove residual prediction.
   - NS `1e-3` physics-only: set `gate_max=0`.
   - NS `1e-4` no growth constraint: set `growth_weight=0`.
   - Shallow-water no curriculum: set one-step and short-rollout epochs to 0.
4. Compare `evaluation_summary.json` values against the promotion gate.
5. Only after all four primary tasks pass, run the joint NS experiment.

## Joint NS Experiment

The fifth experiment trains one model on the existing NS `nu=1e-3` and
`nu=1e-4` files. Each batch has equal counts from both viscosities.

```text
Per viscosity: train 0:999, validation 1200:1399, test 1000:1199
```

Run KNO, iKNO, AM-KNO, original PKNO, and PKNO_v1. Report the two viscosity
test scores separately and their macro average. The claim is limited to
multi-condition training on two seen conditions. It is not an interpolation or
unseen-viscosity result. The unverified `NS_Re5000` Drive files are excluded.

## Checkpoint Rule

V1 model selection uses only validation RL2. The final primary comparison uses
`checkpoint_final.pt` after the fixed 500-epoch schedule and evaluates the test
set once. The old Stage3_0 test-selected checkpoint behavior is not copied.
