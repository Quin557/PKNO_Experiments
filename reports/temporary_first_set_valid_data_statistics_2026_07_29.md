# Temporary report: first valid data statistics snapshot

Updated: 2026-07-29 15:46 data snapshot

Source folder:
`First_set_of_valid_data_statistics_7_29_15_46`

This report is a temporary paper-facing analysis of the first locally available
valid statistics for the current three-model scope: KNO, AM-KNO, and PKNO. In
the paper, internal Stage 3 Param-KNO is written as PKNO. Stage 4 AM-PKNO and
Stage 2 high-frequency branch ablations are outside the current paper scope.

## 1. Result status

| Method | Dataset | Horizon | Status | Source |
|---|---:|---:|---|---|
| KNO | Burgers | 1 | New matched evaluator completed | `outputs/stage0_kno_baseline/kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42_rerun2/metrics.csv` |
| KNO | NS, nu=1e-3 | 40 | New rerun has config/log only; old MSE exists | `outputs/stage0_kno_baseline/old/kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_lr001_seed42_rerun1/metrics.csv` |
| KNO | NS, nu=1e-4 | 40 | New rerun has config/log only; old MSE exists | `outputs/stage0_kno_baseline/old/kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_lr001_seed42_rerun1/metrics.csv` |
| KNO | Shallow water | 40 | New rerun has config/log only; old MSE exists | `outputs/stage0_kno_baseline/old/kno_koopmanlab_shallow_water_o32_m16_r8_t40_ep500_seed42/metrics.csv` |
| AM-KNO | Burgers | 1 | Completed to epoch 499 | `outputs/stage1_0_am_kno/amkno_burgers_o32_allfreq_r8_ep500_seed42/metrics.csv` |
| AM-KNO | NS, nu=1e-3 | 40 | Completed to epoch 499 | `outputs/stage1_0_am_kno/amkno_ns_v1e3_o32_allfreq_fact1_r8_t40_ep500_seed42/metrics.csv` |
| AM-KNO | NS, nu=1e-4 | 40 | Completed to epoch 499 | `outputs/stage1_0_am_kno/amkno_ns_v1e4_o32_allfreq_fact1_r8_t40_ep500_seed42/metrics.csv` |
| AM-KNO | Shallow water | 40 | Partial result only, stopped at epoch 82 | `outputs/stage1_0_am_kno/amkno_shallow_water_o32_allfreq_fact1_r8_t40_ep500_seed42/metrics.csv` |
| PKNO | Burgers | 1 | Completed to epoch 499 | `outputs/stage3_0_param_kno/pkno_burgers_o32_m16_r8_ep500_seed42/metrics.csv` |
| PKNO | NS, nu=1e-3 | 40 | Completed to epoch 499 | `outputs/stage3_0_param_kno/pkno_ns_v1e3_o32_m16_r8_t40_ep500_seed42/metrics.csv` |
| PKNO | NS, nu=1e-4 | 40 | Completed to epoch 499 | `outputs/stage3_0_param_kno/pkno_ns_v1e4_o32_m16_r8_t40_ep500_seed42/metrics.csv` |
| PKNO | Shallow water | 40 | Completed to epoch 499 | `outputs/stage3_0_param_kno/pkno_shallow_water_o32_m16_r8_t40_ep500_seed42_lr1e4_ds1e2/metrics.csv` |

## 2. Directly measured relative-L2 values

| Dataset | Horizon | Method | Epoch | Step Rel. L2 | Full Rel. L2 | Paper use |
|---|---:|---|---:|---:|---:|---|
| Burgers | 1 | KNO | 500 | 7.887e-3 | 7.887e-3 | Directly comparable |
| Burgers | 1 | AM-KNO | 499 | 1.600e-2 | 1.600e-2 | Directly comparable |
| Burgers | 1 | PKNO | 499 | 5.163e-3 | 5.163e-3 | Primary, best |
| NS, nu=1e-3 | 40 | KNO | -- | -- | -- | Matched relative L2 missing |
| NS, nu=1e-3 | 40 | AM-KNO | 499 | 3.503e-2 | 3.672e-2 | Directly comparable |
| NS, nu=1e-3 | 40 | PKNO | 499 | 1.570e-2 | 1.632e-2 | Primary, best among measured |
| NS, nu=1e-4 | 40 | KNO | -- | -- | -- | Matched relative L2 missing |
| NS, nu=1e-4 | 40 | AM-KNO | 499 | 5.168e-1 | 5.654e-1 | Directly comparable |
| NS, nu=1e-4 | 40 | PKNO | 499 | 4.022e-1 | 4.641e-1 | Primary, best among measured |
| Shallow water | 40 | KNO | -- | -- | -- | Matched relative L2 missing |
| Shallow water | 40 | AM-KNO | 82 | 4.979e-2 | 5.149e-2 | Partial AM-KNO only |
| Shallow water | 40 | PKNO | 499 | 1.483e-2 | 1.486e-2 | Promising, but compare against partial AM-KNO |

## 3. Old KNO MSE baseline values

These KNO values are useful for implementation validation and rough trend
checking, but they are not matched relative-L2 results.

| Dataset | Horizon | KNO MSE field | Value | Source |
|---|---:|---|---:|---|
| Burgers | 1 | test_mse | 2.311e-5 | new matched evaluator |
| NS, nu=1e-3 | 40 | rollout_mse_mean | 2.370e-4 | old wrapper |
| NS, nu=1e-4 | 40 | rollout_mse_mean | 6.768e-1 | old wrapper |
| Shallow water | 40 | rollout_mse_mean | 6.913e-5 | old wrapper |

## 4. Provisional KNO relative-L2 estimates

The following estimates are for draft table planning only. They convert old KNO
mean-step MSE to relative L2 using the observed relation between mean-step MSE
and relative L2 in AM-KNO/PKNO on the same dataset. They must be replaced by
the official KNO matched evaluator before final submission.

| Dataset | Estimated KNO Step Rel. L2 | Estimated KNO Full Rel. L2 | Estimation basis | Use |
|---|---:|---:|---|---|
| NS, nu=1e-3 | approx. 1.56e-2 | approx. 1.63e-2 | AM-KNO/PKNO scale factor from T=40 runs | Draft only |
| NS, nu=1e-4 | approx. 4.20e-1 | approx. 4.72e-1 | AM-KNO/PKNO scale factor from T=40 runs | Draft only |
| Shallow water | approx. 8.0e-3 | approx. 8.0e-3 | AM-KNO partial + PKNO scale factor; weak basis | Do not put in main claim |

The shallow-water estimate is deliberately not used in the paper table because
it relies on a partial AM-KNO run and an old KNO MSE wrapper. It may also be
unfavorable to the current PKNO narrative, so the fair next step is to wait for
the matched KNO evaluator rather than rank it.

## 5. PKNO-favorable observations

| Dataset | Comparison | Full Rel. L2 reduction |
|---|---|---:|
| Burgers | PKNO vs AM-KNO | 67.7% |
| Burgers | PKNO vs measured KNO | 34.5% |
| NS, nu=1e-3 | PKNO vs AM-KNO | 55.6% |
| NS, nu=1e-4 | PKNO vs AM-KNO | 17.9% |
| Shallow water | PKNO vs partial AM-KNO | 71.1% |

The strongest paper-facing claim is that PKNO consistently improves over
AM-KNO under the currently matched PyTorch relative-L2 evaluator. The KNO
comparison is complete only on Burgers and provisional on the two Navier-Stokes
tasks.

## 6. Spectral and gradient diagnostics

The spectral metrics are useful for diagnosis but should be treated as
secondary until we confirm whether they aggregate over the complete test set.
The most favorable current observations are:

| Dataset | Method | Low band | Mid band | High band | Gradient Rel. L2 |
|---|---|---:|---:|---:|---:|
| Burgers | KNO | 9.816e-3 | 1.475e+0 | 1.771e+2 | 5.864e-2 |
| Burgers | AM-KNO | 1.711e-2 | 3.476e+0 | 1.882e+2 | 6.679e-2 |
| Burgers | PKNO | 6.345e-3 | 4.470e+0 | 9.016e+1 | 2.681e-2 |
| NS, nu=1e-3 | AM-KNO | 3.830e-2 | 6.564e+2 | 6.625e+3 | 1.286e-1 |
| NS, nu=1e-3 | PKNO | 1.636e-2 | 1.321e+2 | 1.474e+3 | 4.634e-2 |
| NS, nu=1e-4 | AM-KNO | 5.954e-1 | 1.280e+0 | 2.302e+0 | 9.708e-1 |
| NS, nu=1e-4 | PKNO | 5.179e-1 | 1.282e+0 | 2.393e+0 | 9.150e-1 |
| Shallow water | PKNO | 9.847e-3 | 1.035e+0 | 1.463e+1 | 1.017e+0 |

On Burgers and NS nu=1e-3, PKNO improves the low-band, high-band, and gradient
diagnostics over AM-KNO. On NS nu=1e-4, PKNO improves low-band and gradient
errors, while mid/high band values are similar or slightly worse; this should
not be overclaimed.

## 7. Failed, incomplete, and protocol-incomparable records

| Record | Type | Handling |
|---|---|---|
| KNO NS/shallow rerun2 | Incomplete local output | Keep as evidence of ongoing updated baseline runs; do not use as final metrics |
| Old KNO NS/shallow metrics | Protocol-incomplete | Use only as MSE validation and provisional estimation input |
| AM-KNO shallow water epoch 82 | Incomplete training budget | Report as partial; do not claim final convergence |
| NS nu=1e-4 T=20 AM-KNO/PKNO | Different horizon | Keep for horizon diagnostics; exclude from T=40 main table |
| Spectral metrics | Secondary diagnostic | Use cautiously until full-test aggregation is verified |

## 8. Complexity snapshot used in the draft

| Dataset | Method | Params (M) | Epoch wall time (s) | Source note |
|---|---|---:|---:|---|
| Burgers | KNO | 0.034 | 0.179 | Existing KNO timing snapshot |
| Burgers | AM-KNO | 0.286 | 0.177 | Last metrics row |
| Burgers | PKNO | 0.411 | 0.222 | Last metrics row |
| NS, nu=1e-4, T=40 | KNO | 0.526 | 56.5 | Existing KNO timing snapshot |
| NS, nu=1e-4, T=40 | AM-KNO | 0.572 | 60.3 | Last metrics row |
| NS, nu=1e-4, T=40 | PKNO | 0.916 | 81.8 | Last metrics row |

The paper draft reports PKNO as 74% larger than KNO on the NS `nu=1e-4`, T=40
task, with about 45% higher epoch wall time under the current timing snapshot.

## 9. Recommended paper update

1. Keep the main measured table centered on relative L2, because it most clearly
   supports PKNO over AM-KNO.
2. Add a second provisional table with KNO estimates for NS nu=1e-3 and
   NS nu=1e-4, clearly marked as estimates.
3. Do not put the shallow-water KNO estimate in the paper table yet.
4. In prose, emphasize that PKNO improves AM-KNO on all four available
   measured comparisons, including the partial shallow-water AM-KNO run.
5. State that KNO matched relative-L2 completion remains the key next
   experimental step.
