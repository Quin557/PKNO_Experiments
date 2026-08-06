# IKNO and PKNO Baseline Comparison Report

Date: 2026-08-05

Chinese version: [2026_08_05_ikno_pkno_baseline_comparison_zh.md](2026_08_05_ikno_pkno_baseline_comparison_zh.md)

## Scope and data sources

This report consolidates completed runs under:

```text
Latest experimental results uploaded/outputs/
```

The requested comparison contains KNO, IKNO, AM-KNO, PKNO, and the three PKNO-U variants. AM-PKNO is not included in the primary table because it was not requested. All shown completed runs use seed 42 and epoch 499 (500 training epochs), except where noted. Lower error is better.

The primary value is full-rollout relative L2. KNO values come from each run's `evaluation_summary.json`; all other values come from the final row of `metrics.csv`. Burgers, NS v=1e-3, and shallow-water use matched sample counts/splits within their rows. NS v=1e-4 is an exception: official KNO used the KoopmanLab default `ntrain=8000, ntest=200`, while IKNO, AM-KNO, PKNO, and PKNO-U used `ntrain=1000, ntest=200`; their 200 test samples are therefore not the same samples. The NS v=1e-4 KNO values are shown for inventory only and are not a valid direct comparison. T=20 and T=40 are separate experiments and must not be compared across rows.

PKNO-U variants in the uploaded configurations are:

| Variant | `condition_mode` | State embedding |
|---|---|---:|
| PKNO-U A | `physical_only` | 16 |
| PKNO-U B | `physical_compact_state` | 16 |
| PKNO-U C | `physical_gated_state` | 16 |

They are not a pure one-variable ablation of Stage 3 PKNO: in addition to the conditioning route, their state embedding is 16 rather than the Stage 3 PKNO default 64. Interpret them as alternative model variants, not as an isolated gate ablation.

## Main results

Full-rollout relative L2. Bold marks the lowest completed value in that row. Blank cells mean that no completed 500-epoch result exists and are intentionally not filled with partial results.

| Dataset and horizon | KNO | IKNO | AM-KNO | PKNO | PKNO-U A | PKNO-U B | PKNO-U C |
|---|---:|---:|---:|---:|---:|---:|---:|
| Burgers, T=1 | 7.887e-3 | 9.602e-3 | 1.600e-2 | **5.163e-3** | 1.475e-2 | 1.561e-2 | 1.471e-2 |
| NS v=1e-3, T=40 | 1.629e-2 | **1.061e-2** | 3.672e-2 | 1.632e-2 | 5.314e-2 | 4.407e-2 | 4.961e-2 |
| NS v=1e-4, T=20 | 1.798e-1* | **1.597e-1** | 2.493e-1 | 1.694e-1 | 2.749e-1 | 2.679e-1 | 2.759e-1 |
| NS v=1e-4, T=40 | 4.672e-1* | 4.776e-1 | 5.654e-1 | **4.641e-1** | 5.639e-1 | 5.850e-1 | |
| Shallow-water, T=40 | 7.970e-3 | **4.725e-3** | 2.876e-2 | 1.486e-2 | 1.136e-2 | 1.284e-2 | |

`*` denotes the unmatched official-KNO NS v=1e-4 split; do not calculate KNO-versus-other-model improvements from these two cells. PKNO-U C has a stored NS v=1e-4 T=40 value at epoch 420 and a shallow-water value at epoch 343, but neither is a completed run; both are left blank. The earlier Stage 3 PKNO shallow-water run ending at epoch 11 is likewise excluded; the stable `lr=5e-5, decompose=4, delta_scale=0.005` run is used instead.

## Configuration and computational cost

### Measurement conventions and limits

- Every completed run shown here uses 500 epochs and seed 42. The non-KNO `seconds` value is the wall-clock duration of one epoch in `metrics.csv`; it includes that epoch's training and test evaluation. The table reports both the sum over all 500 logged epochs and the final-epoch value. Neither is a pure training-step benchmark.
- The official KNO stage-0 `metrics.csv` does not record comparable epoch wall time. Its `complexity.csv` does record parameter count, peak memory, single-step inference latency, and complete rollout latency. The other implementations did not write peak-memory or inference-latency measurements, so those cells are intentionally blank rather than estimated.
- All timings are environment-dependent. Shallow-water IKNO used batch size 3 after the batch-5 smoke test exhausted a 47.54 GB GPU; the completed AM-KNO, PKNO, and PKNO-U shallow-water runs used batch size 5. Therefore, epoch-time comparisons involving shallow-water IKNO are not batch-normalized.
- The NS v=1e-4 KNO runs use a different 8000/200 split from the 1000/200 split used by the other methods. This affects epoch cost as well as accuracy, so KNO timing in those rows is not protocol-matched.

### Model settings

All models use observable/operator size `O=32`, Fourier modes `modes=16` where a truncated-mode operator is used, Adam with weight decay `1e-4`, `gamma=0.5`, `step_size=100`, and the dataset-specific autoregressive horizon shown below. AM-KNO uses all FFT modes (`max_modes=0`), so its `modes=16` setting is not its operator-frequency restriction.

| Model | Operator/dictionary design | Key architecture hyperparameters | Learning rate by dataset (Burgers / NS 1e-3 / NS 1e-4 / shallow) |
|---|---|---|---|
| KNO | Official KoopmanLab fixed complex matrix per retained Fourier mode | `O=32`, modes 16, `decompose=8` | `1e-3 / 1e-3 / 1e-3 / 1e-3` |
| IKNO | Fixed Fourier Koopman operator plus pointwise invertible residual-coupling dictionary | `O=32`, modes 16, `decompose=4`, `koopman_power=2`, 4 INN blocks, INN hidden width 128 | `1e-3 / 1e-3 / 1e-3 / 1e-3` |
| AM-KNO | Frequency-conditioned operator generator; factorized 2-D operator, rank 1 | `O=32`, all FFT modes, `decompose=8`, generator depth 2, hidden width 128 | `1e-3 / 5e-4 / 3e-4 / 2e-4` |
| PKNO | Shared learned Koopman-matrix dictionary conditioned on physical metadata plus a 64-D state summary | `O=32`, modes 16, dictionary depth 2 / width 128, state embedding 64; `decompose=8` except shallow 4; delta scale `.05` except shallow `.005` | `1e-3 / 5e-4 / 5e-4 / 5e-5` |
| PKNO-U A | PKNO-U, physical metadata only | `O=32`, modes 16, dictionary depth 2 / width 128, state embedding 16, `decompose=8` | `1e-3 / 5e-4 / 3e-4 / 5e-5` |
| PKNO-U B | PKNO-U, physical metadata plus compact state route | Same as A; `condition_mode=physical_compact_state` | `1e-3 / 5e-4 / 3e-4 / 5e-5` |
| PKNO-U C | PKNO-U, physical metadata plus gated state route | Same as A; `condition_mode=physical_gated_state` | `1e-3 / 5e-4 / 3e-4 / 5e-5` |

The data protocol is `ntrain/ntest=1000/200` for Burgers and both Navier-Stokes viscosity settings in the shared-trainer models. Shallow-water is `900/100`, with 128 x 128 spatial fields, `T_in=10`, and `T_out=40`. KNO uses its official default `8000/200` NS v=1e-4 protocol, as noted above. Burgers is a one-step rollout (`T=1`); the listed Navier-Stokes and shallow-water configurations use `T_in=10` and the stated `T_out` horizon.

### Parameter count and observed training time

Parameter counts are exact values written by the corresponding completed runs. `500 epochs total (h) / final epoch (s)` is respectively the sum of the 500 `seconds` records, converted to hours, and the final-row `seconds` value. It includes train and evaluation time. A blank cell indicates that the run did not record that measurement; it does not mean zero cost. PKNO-U C is blank for its incomplete T=40 and shallow-water runs.

| Dataset and horizon | Measurement | KNO | IKNO | AM-KNO | PKNO | PKNO-U A | PKNO-U B | PKNO-U C |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Burgers, T=1 | Parameters | 33,921 | 116,832 | 286,242 | 411,001 | 469,305 | 474,453 | 474,470 |
|  | 500 epochs total (h) / final epoch (s) | | 0.042 / 0.302 | 0.025 / 0.179 | 0.033 / 0.222 | 0.090 / 0.659 | 0.087 / 0.603 | 0.085 / 0.642 |
| NS v=1e-3, T=40 | Parameters | 526,026 | 608,352 | 571,883 | 915,585 | 1,135,553 | 1,147,721 | 1,147,738 |
|  | 500 epochs total (h) / final epoch (s) | | 9.655 / 68.224 | 9.580 / 67.818 | 14.004 / 93.185 | 27.465 / 192.557 | 29.402 / 223.273 | 29.876 / 200.696 |
| NS v=1e-4, T=20 | Parameters | 526,026 | 608,352 | 571,883 | 915,585 | 1,135,553 | 1,147,721 | 1,147,738 |
|  | 500 epochs total (h) / final epoch (s) | | 4.996 / 36.369 | 4.334 / 54.657 | 7.294 / 53.860 | 14.046 / 98.051 | 14.498 / 106.184 | 14.659 / 110.645 |
| NS v=1e-4, T=40 | Parameters | 526,026 | 608,352 | 571,883 | 915,585 | 1,135,553 | 1,147,721 | |
|  | 500 epochs total (h) / final epoch (s) | | 9.687 / 68.990 | 9.189 / 60.261 | 12.765 / 81.822 | 29.025 / 209.556 | 29.013 / 230.714 | |
| Shallow-water, T=40 | Parameters | 526,026 | 608,352 | 571,883 | 915,585 | 1,135,553 | 1,147,721 | |
|  | 500 epochs total (h) / final epoch (s) | | 31.184 / 220.730 (batch 3) | 26.168 / 186.569 (batch 5) | 16.929 / 109.266 (batch 5) | 53.266 / 388.030 (batch 5) | 55.604 / 398.075 (batch 5) | |

### Available KNO inference and memory measurements

These are the only directly recorded deployment-time measurements in the uploaded results. `Rollout (ms)` is one full autoregressive evaluation over the stated horizon, while `Step (ms)` is the associated per-step measurement. The parameter counts correspond to the KNO column above.

| KNO dataset and horizon | Peak GPU memory (GB) | Step (ms) | Rollout (ms) |
|---|---:|---:|---:|
| Burgers, T=1 | 0.03224 | 2.108 | 2.108 |
| NS v=1e-3, T=40 | 0.15244 | 3.090 | 123.602 |
| NS v=1e-4, T=20 | 0.10999 | 2.752 | 55.037 |
| NS v=1e-4, T=40 | 0.15244 | 3.076 | 123.050 |
| Shallow-water, T=40 | 0.27408 | 3.667 | 146.677 |

### Cost interpretation

IKNO is relatively compact: it adds about 82k parameters over KNO in 1-D and about 82k in 2-D, while achieving the best completed accuracy on NS v=1e-3 and shallow-water. The current PKNO has 74% more parameters than KNO in 2-D (915,585 versus 526,026); the PKNO-U variants have about 116-118% more. Their final-epoch times are correspondingly generally higher, especially on the T=40 Navier-Stokes runs. This cost is not justified by a uniform accuracy gain over KNO in the current single-condition setup.

For shallow-water, the observed PKNO final epoch is faster than AM-KNO despite having more parameters. This should be treated as an implementation-path observation, not a general complexity claim: AM-KNO's all-frequency generator and factorized operator have a different computation pattern, and these numbers include data loading and evaluation. A fair efficiency result needs a shared benchmark that records warm-up-discarded training throughput, peak allocated memory, and batched inference latency for every model under identical batch size, precision, device, and test subset.

## What the table supports

### IKNO

IKNO is the strongest completed non-KNO result on three of five rows: NS v=1e-3 T=40, NS v=1e-4 T=20, and shallow-water T=40. Its shallow-water relative L2 is 40.7% lower than KNO. It is not uniformly dominant: it is worse than KNO on Burgers. The NS v=1e-4 KNO comparison is not valid because of the split mismatch.

### PKNO versus AM-KNO

PKNO improves on AM-KNO in every matched row: 67.7% on Burgers, 55.6% on NS v=1e-3 T=40, 32.1% on NS v=1e-4 T=20, 17.9% on NS v=1e-4 T=40, and 48.3% on shallow-water T=40. This is a real and internally consistent result: the shared dictionary plus conditioned Fourier Koopman construction is substantially better than the current frequency-only AM-KNO baseline.

### PKNO versus KNO

The stronger claim, that PKNO is broadly superior to official KNO for long rollout, is not established.

- Burgers: PKNO is 34.5% better than KNO, but this is a one-step task and does not test long-horizon stability.
- NS v=1e-3 T=40: PKNO and KNO are effectively tied in full relative L2 (1.632e-2 versus 1.629e-2).
- NS v=1e-4 T=20 and T=40: the present KNO comparison is invalid because KNO used 8000 training samples and a different 200-sample test segment, whereas the other models used 1000/200. No KNO-versus-PKNO claim can be made from these rows.
- Shallow-water T=40: PKNO is 86.5% worse than KNO, although much better than AM-KNO.

Therefore, the current results support a narrower claim: PKNO repairs much of the loss introduced by AM-KNO and is competitive with KNO on NS v=1e-3, but it does not yet demonstrate a reliable, large long-horizon advantage over KNO.

## Long-rollout diagnostic

The table below uses the first and final rows of `rollout_error_by_step.csv`. These are per-step MSE values, so they should only be compared within the same dataset and horizon. The growth factor is final-step MSE divided by first-step MSE.

| Dataset | Model | Step 1 MSE | Final-step MSE | Growth |
|---|---|---:|---:|---:|
| NS v=1e-3, T=40 | KNO | 7.282e-5 | 6.059e-4 | 8.32x |
|  | IKNO | 1.344e-4 | **2.238e-4** | **1.67x** |
|  | PKNO | 1.713e-4 | 5.170e-4 | 3.02x |
| NS v=1e-4, T=40 | KNO | 3.575e-2 | **1.876e+0** | **52.48x** |
|  | IKNO | 3.303e-2 | 2.491e+0 | 75.42x |
|  | PKNO | **2.125e-2** | 2.076e+0 | 97.68x |
| Shallow-water, T=40 | KNO | **3.625e-5** | 1.167e-4 | 3.22x |
|  | IKNO | 3.535e-5 | **4.505e-5** | **1.27x** |
|  | PKNO | 2.063e-4 | 3.188e-4 | 1.55x |

There is one positive PKNO long-rollout signal: on NS v=1e-3, its first-step MSE is worse than KNO, but its final-step MSE is 14.7% lower and its growth factor is lower. The full-rollout relative L2 remains tied because PKNO gives away too much accuracy in the early part of the rollout.

The other two T=40 cases do not establish the desired behavior. On NS v=1e-4, PKNO's error grows 97.68x from its own first to final step, but the displayed KNO curve is from a different split and cannot be used as the comparison. On shallow-water, PKNO has a low growth factor but starts about 5.7 times worse than KNO; stable propagation cannot compensate for the inaccurate initial prediction.

## Why PKNO has not produced a large rollout gain

1. **The physical condition is constant within each current training run.** The Stage 3 loaders repeat one viscosity/grid/time condition vector for every sample in a single dataset file. Consequently, PKNO is not trained on a family in which viscosity or another physical parameter varies across samples. Its parameterized Koopman map has little opportunity to learn a useful physical-condition response; it can only exploit the state summary computed from the history.

2. **The objective does not explicitly prioritize late-time accuracy or stability.** The common trainer sums prediction MSE across rollout steps. It has no larger late-step weight, final-state term, rollout-growth penalty, spectral-radius constraint, conservation term, or latent Koopman-consistency loss. A model can obtain a competitive average loss while still accumulating error late in the rollout.

3. **The main comparison is against a strong, specialized KNO.** KNO already has a fixed per-frequency complex matrix tuned independently for the one dataset being trained. PKNO replaces that direct freedom with a shared dictionary and a conditioned operator generator. When conditions do not vary across samples, this added structure can be an optimization burden rather than additional information. The current KNO NS v=1e-4 runs also use a different 8000-sample protocol, so they are not a valid PKNO baseline until rerun with the 1000/200 split.

4. **The present conditioning variants do not solve the issue.** PKNO-U A/B/C are consistently worse than Stage 3 PKNO on their completed rows. B is generally better than A on Navier-Stokes, indicating that state information can help, but neither matches Stage 3 PKNO or IKNO on the completed matched rows. This argues against presenting the current conditioned/gated routes as the source of a validated improvement.

5. **Evidence remains single-seed and mixed-implementation.** Every result is seed 42. KNO is evaluated through its official wrapper while the other models use the shared PyTorch rollout trainer. The scalar full relative L2 values are useful, but small margins are not publishable claims without repeated seeds and a unified evaluator check.

## Is the current PKNO result satisfactory?

It is satisfactory as an intermediate ablation result, not as the intended headline result.

The model meets the weaker objective of materially outperforming the current AM-KNO implementation and has a promising NS v=1e-3 error-growth signature. It does not meet the stronger objective of a clear, repeatable long-rollout advantage over KNO across difficult PDEs. The shallow-water result is the clearest counterexample; NS v=1e-4 requires a matched KNO rerun before it can be used as supporting or opposing evidence.

## Recommended next experiments

1. Train a genuinely joint Navier-Stokes model over mixed viscosities (at minimum v=1e-3 and v=1e-4 in the same train set), retain viscosity in the condition vector, and test interpolation/extrapolation by viscosity. This is the experiment that can actually test a parameterized Koopman family.

2. Add a rollout-aware objective: increasing weights toward late steps, a final-step relative L2 term, and a stability regularizer on generated Koopman matrices. Report both mean and final-step relative L2.

3. Run a controlled PKNO ablation in which only the conditioning route changes. Keep observable dimension, dictionary depth/hidden width, state embedding width, optimizer, horizon, and seed set identical.

4. First rerun official KNO on NS v=1e-4 with explicit `--ntrain 1000 --ntest 200`, then rerun KNO, IKNO, PKNO, and the selected PKNO-U variant with at least three seeds under one evaluator. Report mean plus standard deviation and retain the per-step error curves.

5. Before emphasizing high-frequency claims, add full-test-set spectral metrics rather than the current first-evaluation-batch statistic.
