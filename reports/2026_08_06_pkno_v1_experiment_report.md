# PKNO_v1 Latest Experiment Report

Date: 2026-08-06  
Results: `Latest experimental results uploaded/outputs/`  
Primary model: `stage3_2_pkno_v1`, seed 42, 500 epochs, `O=32`, `modes=16`

## Executive summary

PKNO_v1 is an independent route. It keeps the Fourier Koopman operator, shared dictionary, and 1x1 skip convolution; it does not use a U-Net. The main changes are physics-first conditioning with bounded state modulation, physical-field residual prediction, a soft growth-envelope penalty for NS `nu=1e-4`, and a `1 -> 5 -> 10 -> 40` rollout curriculum.

MSE remains the training objective, with reconstruction and stability regularizers. RL2 is used for validation, reporting, and comparison only.

PKNO_v1 beats the old PKNO on NS `nu=1e-4,T=40` and shallow-water, and also improves NS `nu=1e-4,T=20`. It does not beat the old PKNO on Burgers or NS `nu=1e-3,T=40`. Therefore it fails the required four-task promotion gate and must not replace the paper PKNO result yet.

## Protocol and caveats

- All primary results are complete 500-epoch runs with seed 42.
- Burgers uses train/test 1000/200 and one-step prediction.
- Both NS conditions use 1000/200 for PKNO, iKNO, AM-KNO, AM-PKNO, and PKNO_v1. PKNO_v1 additionally uses 1200:1399 as validation while keeping test 1000:1199.
- Shallow-water uses train/test 900/100.
- The actual old PKNO shallow-water run uses `decompose=4`, while the PKNO_v1 shallow-water run uses `decompose=8`. This is not a strict same-configuration comparison and must be rerun with `decompose=4`.
- Official KNO uses 8000/200 for NS `nu=1e-4`, unlike the other models. Those values are marked `*` and are reference-only.

## Full-rollout RL2

Lower is better.

| Task | KNO | iKNO | AM-KNO | Old PKNO | AM-PKNO | PKNO_v1 |
|---|---:|---:|---:|---:|---:|---:|
| Burgers, T=1 | 7.887e-3 | 9.602e-3 | 1.600e-2 | **5.163e-3** | 1.265e-2 | 7.694e-3 |
| NS `nu=1e-3`, T=40 | 1.629e-2 | **1.061e-2** | 3.672e-2 | 1.632e-2 | 3.030e-2 | 1.714e-2 |
| NS `nu=1e-4`, T=20 | 1.798e-1* | **1.597e-1** | 2.493e-1 | 1.694e-1 | 2.161e-1 | 1.602e-1 |
| NS `nu=1e-4`, T=40 | 4.672e-1* | 4.776e-1 | 5.654e-1 | 4.641e-1 | 5.313e-1 | **4.140e-1** |
| Shallow-water, T=40 | 7.970e-3 | **4.725e-3** | 2.876e-2 | 1.486e-2 | 2.101e-2 | 1.230e-2 |

Relative to old PKNO, PKNO_v1 is 49.0% worse on Burgers, 5.1% worse on NS `1e-3,T=40`, 5.4% better on NS `1e-4,T=20`, 10.8% better on NS `1e-4,T=40`, and 17.2% better on shallow-water.

## Rollout and spectral evidence

For NS `nu=1e-4`, PKNO_v1 has lower field MSE than old PKNO at steps 1, 5, 10, 20, 30, and 39: `1.29e-2, 5.66e-2, 1.60e-1, 5.12e-1, 8.54e-1, 1.62` versus `3.61e-2, 9.56e-2, 2.11e-1, 5.73e-1, 1.07, 2.08`. This is consistent with the intended long-horizon stability effect.

The spectral diagnostics also support the mechanism. On NS `nu=1e-4`, high-band spectral RL2 drops from `2.393` to `1.284`, HF energy-ratio error from `7.55e-5` to `4.66e-6`, and gradient RL2 from `0.915` to `0.840`. On shallow-water, high-band spectral RL2 drops from `14.63` to `1.07`, HF energy-ratio error from `9.93e-5` to `3.92e-7`, and gradient RL2 from `1.017` to `0.527`.

The failure modes are also informative. Burgers PKNO_v1 high-band spectral RL2 is `591.1` versus `90.2` for old PKNO, with gradient RL2 `0.0648` versus `0.0268`. For NS `nu=1e-3`, V1 is slightly better during the first 10 steps but becomes worse after about step 20, indicating residual long-horizon drift rather than a one-step failure.

## Parameters and wall-clock cost

| Model | Burgers parameters | NS / 2-D-field parameters |
|---|---:|---:|
| KNO | 33,921 | 526,026 |
| iKNO | 116,832 | 608,352 |
| AM-KNO | 286,242 | 571,883 |
| Old PKNO | 411,001 | 915,585 |
| PKNO_v1 | 444,025 | 948,353 |

PKNO_v1 adds about 8.0% parameters on Burgers and 3.6% on the 2-D tasks relative to old PKNO.

| Task | Old PKNO total | PKNO_v1 total | Old avg/last epoch | V1 avg/last epoch |
|---|---:|---:|---:|---:|
| Burgers | 0.03 h | 0.04 h | 0.24 / 0.22 s | 0.28 / 0.25 s |
| NS `1e-3`, T=40 | 14.00 h | 10.42 h | 100.83 / 93.18 s | 74.99 / 81.16 s |
| NS `1e-4`, T=20 | 7.29 h | 6.13 h | 52.52 / 53.86 s | 44.11 / 51.23 s |
| NS `1e-4`, T=40 | 12.76 h | 10.64 h | 91.90 / 81.82 s | 76.64 / 86.76 s |
| Shallow-water, T=40 | 16.93 h | 18.01 h | 121.89 / 109.27 s | 129.65 / 147.62 s |

These are not pure speed comparisons: V1 uses short-rollout warm-up and NS validation, while old PKNO starts with full rollout. V1 peak GPU memory and inference latency are not present in the uploaded artifacts, so no memory or inference-speed claim is justified.

## Training and stability checks

All completed V1 full runs finished without NaN, Inf, OOM, or traceback. NS `nu=1e-4` uses a nonzero growth weight and reports a growth ceiling around 1.09. The final summaries use `checkpoint=final`; validation-best checkpoints were not evaluated on test because checkpoint files are absent from the uploaded result directory. The best validation RL2 is approximately `1.693e-2` at epoch 488 for NS `1e-3` and `4.195e-1` at epoch 492 for NS `1e-4`.

## Conclusions and next actions

1. The strongest evidence is the low-viscosity NS result: V1 improves full T=40 RL2 by 10.8% and improves rollout, spectral, gradient, and HF-energy diagnostics.
2. Shallow-water also improves, but the actual `decompose` mismatch prevents a strict claim until a `decompose=4` rerun.
3. Burgers and NS `1e-3` remain regressions, so V1 is a partially successful research branch, not a paper replacement.
4. First rerun shallow-water with `decompose=4` and add peak-memory/inference instrumentation. Then run minimal ablations on NS `1e-3` and Burgers: physics-only (`gate_max=0`), direct prediction, and no curriculum. Keep `modes=16` fixed.
5. Retain the NS `1e-4` stability direction, but confirm it with additional seeds before any paper update. Run the joint NS experiment only after all four primary tasks pass under a matched protocol.

## Reproducibility files

- Model design: [docs/models/stage3_2_pkno_v1_model_design.md](../docs/models/stage3_2_pkno_v1_model_design.md)
- Experiment design: [docs/experiments/stage3_2_pkno_v1_experiment_design.md](../docs/experiments/stage3_2_pkno_v1_experiment_design.md)
- PKNO_v1 outputs: `Latest experimental results uploaded/outputs/stage3_2_pkno_v1/`
- Old PKNO outputs: `Latest experimental results uploaded/outputs/stage3_0_param_kno/`
