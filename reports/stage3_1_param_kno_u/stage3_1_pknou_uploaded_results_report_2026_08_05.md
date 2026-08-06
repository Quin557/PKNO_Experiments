# Stage3_1 PKNO-U Uploaded Results Report

Date: 2026-08-05

## Scope and evidence

This report analyzes the completed runs in
`Latest experimental results uploaded/outputs/`.  It covers KNO, iKNO,
AM-KNO, PKNO, AM-PKNO, and the three Stage3_1 PKNO-U variants:

| Variant | `condition_mode` | Intended role |
|---|---|---|
| PKNO-U A | `physical_only` | Main model: Koopman operator depends only on explicit physical conditions. |
| PKNO-U B | `physical_compact_state` | Ablation: add a compact history-derived state condition. |
| PKNO-U C | `physical_gated_state` | Ablation: gate the compact state condition. |

All completed runs use seed 42 and 500 training epochs (final CSV row:
`epoch=499`).  Unless noted, the primary metric is final full-rollout
relative L2 (RL2), where lower is better.  Values for PyTorch routes come
from the final row of `metrics.csv`; KNO RL2 comes from its
`evaluation_summary.json`.

Two comparability restrictions are material:

1. Official KNO NS `nu=1e-4` runs used `ntrain=8000, ntest=200`, while the
   PyTorch routes used `ntrain=1000, ntest=200`.  The test segments are not
   identical.  KNO is retained for inventory, but not used to claim an
   improvement on these two rows.
2. The stable PKNO shallow-water run uses `decompose=4`, while PKNO-U A/B use
   `decompose=8`.  The shallow-water gain cannot be attributed to U-Net alone.
   It may include an effect of the different decomposition depth.

PKNO-U C is incomplete for NS `nu=1e-4`, T=40 and shallow water.  The partial
outputs at epochs 420 and 343 are intentionally excluded.

AM-PKNO NS `nu=1e-4`, T=20 is also not an available result in this upload.  Its
output directory contains `args.json`, `config.yaml`, and `env.txt`, but
`metrics.csv` is zero bytes and the corresponding log contains only
`nohup: ignoring input`; there is no checkpoint, rollout CSV, or evaluation
summary anywhere else in this workspace.  The `--` cell is therefore not a
parser omission.  The configuration points to the T=20 dataset, but no trained
metric was uploaded.

## Accuracy: cross-model comparison

Full-rollout RL2.  Bold denotes the lowest completed value among routes with
the same displayed protocol.  Asterisks identify the unmatched KNO NS
`nu=1e-4` protocol.

| Dataset and horizon | KNO | iKNO | AM-KNO | PKNO | AM-PKNO | PKNO-U A | PKNO-U B | PKNO-U C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Burgers, T=1 | 7.887e-3 | 9.602e-3 | 1.600e-2 | **5.163e-3** | 1.265e-2 | 1.475e-2 | 1.561e-2 | 1.471e-2 |
| NS, nu=1e-3, T=40 | 1.629e-2 | **1.061e-2** | 3.672e-2 | 1.632e-2 | 3.030e-2 | 5.314e-2 | 4.407e-2 | 4.961e-2 |
| NS, nu=1e-4, T=20 | 1.798e-1* | **1.597e-1** | 2.493e-1 | 1.694e-1 | -- | 2.749e-1 | 2.679e-1 | 2.759e-1 |
| NS, nu=1e-4, T=40 | 4.672e-1* | 4.776e-1 | 5.654e-1 | **4.641e-1** | 5.313e-1 | 5.639e-1 | 5.850e-1 | incomplete |
| Shallow water, T=40 | 7.970e-3 | **4.725e-3** | 2.876e-2 | 1.486e-2 | 2.101e-2 | 1.136e-2 | 1.284e-2 | incomplete |

### Direct PKNO-U ablation

The table below shows the direct Stage3 baseline and the changes caused by the
three Stage3_1 alternatives.  Percentages use RL2; negative means lower error.

| Dataset and horizon | PKNO | A vs. PKNO | B vs. A | C vs. A | Interpretation |
|---|---:|---:|---:|---:|---|
| Burgers, T=1 | 5.163e-3 | 1.475e-2 (+185.7%) | +5.8% | -0.2% | None of A/B/C is competitive with PKNO. |
| NS, nu=1e-3, T=40 | 1.632e-2 | 5.314e-2 (+225.7%) | -17.1% | -6.6% | Compact state helps A, but remains far behind PKNO. |
| NS, nu=1e-4, T=20 | 1.694e-1 | 2.749e-1 (+62.3%) | -2.6% | +0.4% | B has a small gain over A only. |
| NS, nu=1e-4, T=40 | 4.641e-1 | 5.639e-1 (+21.5%) | +3.7% | incomplete | A is better than B, but both are worse than PKNO. |
| Shallow water, T=40 | 1.486e-2 | 1.136e-2 (-23.6%) | +13.1% | incomplete | A improves, but depth differs (4 vs. 8). |

## Model capacity and implementation differences

The parameter count is taken from each run's `env.txt` / final `metrics.csv`.
The 2D count applies to all NS and shallow-water runs within the method.

| Method | Parameters: Burgers / 2D | Decompose | Modes | Conditioning or adaptation | Main added mechanism |
|---|---:|---:|---|---|---|
| KNO | 0.034M / 0.526M | 8 | 16 | none | Fixed mode-indexed Koopman operator. |
| iKNO | 0.117M / 0.608M | 4 | 16 | none | iKNO with `p=2`. |
| AM-KNO | 0.286M / 0.572M | 8 | all-frequency | frequency adaptive | Frequency-generated operator. |
| PKNO | 0.411M / 0.916M | 8; SWE 4 | 16 | physical parameterization | Shared dictionary and parameterized operator. |
| AM-PKNO | 0.382M / 0.697M | 8; SWE 4 | all-frequency | adaptive parameterization | AM-PKNO route. |
| PKNO-U A | 0.469M / 1.136M | 8 | 16 | physical only | Stable operator norm <= 0.98 and four late-layer latent U-Nets. |
| PKNO-U B | 0.474M / 1.148M | 8 | 16 | physical + compact state | Same U-Net/stability route as A. |
| PKNO-U C | 0.474M / 1.148M | 8 | 16 | gated physical + state | Same U-Net/stability route as A. |

PKNO-U A adds only 14.2% parameters on Burgers and 24.0% on 2D data versus
PKNO.  B/C add about 1% beyond A.  Therefore, their large runtime change is
not explained by parameter count alone; U-Net feature-map computation is the
dominant extra cost.

## Training configuration

All logged routes use 500 epochs and a StepLR schedule with `step_size=100`
and `gamma=0.5`.  Weight decay is `1e-4` for the modern PyTorch routes.  PKNO
and PKNO-U use rollout prediction MSE plus reconstruction MSE; PKNO-U's logged
objective is `5 * prediction_MSE + 0.5 * reconstruction_MSE`.  RL2 is an
evaluation metric, not the training loss.

| Dataset and horizon | Batch size | PKNO LR | PKNO-U LR | Gradient clip: PKNO / U | Decompose: PKNO / U | Comparability |
|---|---:|---:|---:|---:|---:|---|
| Burgers, T=1 | 64 | 1e-3 | 1e-3 | none / none | 8 / 8 | Strict architecture comparison. |
| NS, nu=1e-3, T=40 | 10 | 5e-4 | 5e-4 | 1.0 / 1.0 | 8 / 8 | Strict architecture comparison. |
| NS, nu=1e-4, T=20 | 10 | 5e-4 | 3e-4 | 1.0 / 1.0 | 8 / 8 | U-Net and learning-rate changes are confounded. |
| NS, nu=1e-4, T=40 | 10 | 5e-4 | 3e-4 | 1.0 / 1.0 | 8 / 8 | U-Net and learning-rate changes are confounded. |
| Shallow water, T=40 | 5 | 5e-5 | 5e-5 | 0.1 / 0.1 | 4 / 8 | U-Net and decomposition depth are confounded. |

The present A/B/C comparison is also not a pure conditioning ablation of
Stage3 PKNO: its state embedding size is 16, while the earlier Stage3 PKNO
route used a state embedding size of 64.  Future claims should compare routes
with the same dictionary, operator generator, state width, optimizer, and
training protocol.

## Computational performance

`seconds` is the measured wall-clock time per epoch.  Total time is
`seconds * 500 / 3600`, rounded below.  Runs were logged on NVIDIA RTX A6000
devices with matching Python/PyTorch/CUDA versions, but different device IDs;
server load can affect small differences.

| Dataset and horizon | AM-KNO | AM-PKNO | iKNO | PKNO | PKNO-U A | PKNO-U B | PKNO-U C |
|---|---:|---:|---:|---:|---:|---:|---:|
| Burgers, T=1 | 0.025 h | 0.043 h | 0.042 h | 0.031 h | 0.091 h | 0.084 h | 0.089 h |
| NS, nu=1e-3, T=40 | 9.42 h | 28.13 h | 9.48 h | 12.94 h | 26.74 h | 31.01 h | 27.87 h |
| NS, nu=1e-4, T=20 | 7.59 h | -- | 5.05 h | 7.48 h | 13.62 h | 14.75 h | 15.37 h |
| NS, nu=1e-4, T=40 | 8.37 h | 28.15 h | 9.58 h | 11.36 h | 29.11 h | 32.04 h | incomplete |
| Shallow water, T=40 | 25.91 h | 65.37 h | 30.66 h | 15.18 h | 53.89 h | 55.29 h | incomplete |

| Dataset and horizon | PKNO epoch time | PKNO-U A epoch time | A / PKNO |
|---|---:|---:|---:|
| Burgers, T=1 | 0.222 s | 0.659 s | 2.96x |
| NS, nu=1e-3, T=40 | 93.2 s | 192.6 s | 2.07x |
| NS, nu=1e-4, T=20 | 53.9 s | 98.1 s | 1.82x |
| NS, nu=1e-4, T=40 | 81.8 s | 209.6 s | 2.56x |
| Shallow water, T=40 | 109.3 s | 388.0 s | 3.55x |

The official KNO evaluator is the only route that exported inference and peak
memory measurements.  These values are useful as an inventory but are not a
cross-model efficiency ranking because the other routes did not log matching
measurements.

| KNO task | Peak memory | Inference / step | Full rollout inference |
|---|---:|---:|---:|
| Burgers, T=1 | 0.032 GB | 2.108 ms | 2.108 ms |
| NS, nu=1e-3, T=40 | 0.152 GB | 3.090 ms | 123.602 ms |
| NS, nu=1e-4, T=20 | 0.110 GB | 2.752 ms | 55.037 ms |
| NS, nu=1e-4, T=40 | 0.152 GB | 3.076 ms | 123.050 ms |
| Shallow water, T=40 | 0.274 GB | 3.667 ms | 146.677 ms |

PKNO, iKNO, AM-KNO, AM-PKNO, and PKNO-U do not record peak GPU memory,
per-step inference latency, or full-rollout inference latency.  Consequently,
no cross-model memory-efficiency or deployment-latency claim is supported.
`checkpoint_unet=true` confirms activation checkpointing was used in PKNO-U,
but it does not substitute for a peak-memory measurement.

## Ablation-specific interpretation

### A: physical-only conditioning

A is the correct conceptual main line because the generated Koopman operator
depends only on explicit physical conditions, preserving an interpretable fixed
operator during a rollout.  However, it loses to PKNO on every strict matched
comparison: +185.7% RL2 on Burgers and +225.7% on NS `nu=1e-3`, T=40.  The NS
`nu=1e-4` results also degrade despite a contractive operator.  Thus, the
stability constraint prevents numerical operator explosion but does not recover
the baseline's predictive accuracy.

The shallow-water improvement is promising but preliminary.  A reduces RL2 by
23.6%, prediction MSE from `9.587e-3` to `5.606e-3`, and gradient RL2 from
1.017 to 0.517.  It is the only completed task that supports a high-frequency
benefit.  Because `decompose` differs, it must be confirmed by matched r=4 and
r=8 controls before being presented as a U-Net result.

### B: compact state conditioning

B improves upon A on NS `nu=1e-3`, T=40 by 17.1% and NS `nu=1e-4`, T=20 by
2.6%.  This is evidence that a low-dimensional state summary can sometimes add
useful local information.  It is not a general solution: B is worse than A on
Burgers (+5.8%), NS `nu=1e-4`, T=40 (+3.7%), and shallow water (+13.1%).  B is
also the slowest completed PKNO-U variant on the long NS runs.

The likely structural issue is that B makes the operator condition depend on
the rolling predicted history.  This removes the fixed physical operator
interpretation and lets state-estimation error enter the transition generator at
every rollout step.  Its mixed performance is consistent with this tradeoff.

### C: gated state conditioning

On its completed tasks C is marginally better than A on Burgers (0.2%) and
NS `nu=1e-3`, T=40 (6.6%), while it is 0.4% worse on NS `nu=1e-4`, T=20.  It
does not produce enough evidence for a gated-state conclusion, because the two
most informative remaining long-horizon tasks are incomplete.

The diagnostics record a mean `condition_gate` of approximately 0.5 for the
completed C runs.  This does not prove that the gate is broken, but it provides
no evidence that it learned a meaningful time- or state-dependent balance.
Inspect gate distributions over samples and rollout steps before investing in a
full C sweep.

## Stability and high-frequency diagnostics

Across completed PKNO-U runs, the maximum logged operator spectral norm lies
between 0.877 and 0.977, below the configured 0.98 cap.  Latent RMS remains
bounded during the recorded rollouts.  The new transition is therefore not
visibly exploding, but bounded latent dynamics alone does not imply accurate
long-term prediction.

Gradient RL2 supports the same conclusion.  Relative to PKNO, A is worse on
Burgers (0.1046 vs. 0.0268), NS `nu=1e-3` (0.1292 vs. 0.0463), NS `nu=1e-4`,
T=20 (0.7865 vs. 0.6142), and T=40 (0.9583 vs. 0.9150).  Only shallow water
improves (0.5175 vs. 1.0174).

There are two diagnostic limitations that must be resolved before making a
high-frequency claim:

1. `spectral_metrics.csv` covers only the first evaluation batch.  Its
   high-band relative errors can be ill-conditioned when true high-frequency
   energy is close to zero, so they are not paper-level aggregate metrics.
2. `unet_highpass_rms` is recorded as zero in every completed PKNO-U run.  This
   does not by itself prove the U-Net residual is inactive, but the current log
   cannot demonstrate a nonzero high-frequency contribution.  Treat this as a
   priority instrumentation check.

## Recommended next experiments

### Priority 0: validate the U-Net path before more full runs

1. Add per-U-Net-layer RMS for raw output, high-pass output, and the scaled
   residual actually added to the latent state.  Log these over the full test
   set, not one batch.
2. Log per-layer gradient norms and parameter update norms for every U-Net.
   A zero/near-zero high-pass path, or an inactive gradient path, invalidates a
   U-Net ablation regardless of final RL2.
3. Add `torch.cuda.max_memory_allocated`, `max_memory_reserved`, one-step
   inference time, and full T=40 rollout time to every output directory.

### Priority 1: make the U-Net conclusion causal

1. Run an exact PKNO-U architecture control with `hf_residual_scale=0` (or no
   U-Net modules), using the same stable operator, dictionary, optimizer,
   state width, seed, and `decompose`.  This isolates the U-Net contribution.
2. On shallow water, run both PKNO and PKNO-U at r=4 and r=8.  The resulting
   2x2 comparison separates the U-Net effect from decomposition depth.
3. On NS `nu=1e-4`, repeat A with the PKNO learning rate of `5e-4`; only then
   attribute any remaining difference to the architecture.
4. Complete C only after the instrumentation check passes.  Its two missing
   T=40/shallow-water results are necessary, but finishing them without
   verifying the gate/U-Net path is lower-value compute.

### Priority 2: redesign the parameterization experiment

The physical condition vector is constant across samples inside each current
dataset file.  A single-file run therefore does not genuinely test whether
`c_n` learns a family of physical operators.  Train one joint model with mixed
viscosities at minimum (`nu=1e-3` and `nu=1e-4`), then test held-out
interpolation and extrapolation in viscosity.

For long-horizon stability, keep the operator generator conditioned on the
explicit physical vector only.  If state information is needed, use a compact
state code in a decoder correction or an initial latent adaptation, rather than
feeding the autoregressive state directly into the Koopman matrix at every
step.  If a dynamic state-conditioned operator is retained, constrain its
increment and temporally filter the state code; otherwise prediction error can
change the transition itself over time.

### Priority 3: improve the objective and evaluation protocol

Keep MSE as the primary training objective for continuity with existing runs.
Do not replace it with RL2 loss.  Instead, evaluate a controlled auxiliary
ablation with a late-step MSE weight, final-step MSE term, or gradient/spectral
loss.  Report full-rollout RL2, per-step error curves, final-step error,
full-test-set spectral metrics, runtime, memory, and mean/std over at least
three seeds.

## Decision summary

The current evidence does not justify replacing PKNO with PKNO-U as the default
model.  PKNO-U A has one promising but confounded shallow-water result; B/C do
not provide a reliable state-conditioning improvement; and all variants cost
substantially more compute.  The immediate research value is to validate the
high-pass U-Net path and run a strictly controlled no-U-Net/decompose ablation.
Only after that should the parameterization design be judged using a genuinely
multi-physics training family.
