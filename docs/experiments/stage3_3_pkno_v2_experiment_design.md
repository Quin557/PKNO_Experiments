# Stage3_3 PKNO_v2-A Experiment Design / 实验设计

## Main four experiments / 四项主实验

| ID | Dataset | Split | T_in/T_out | decompose | Default lr |
|---|---|---:|---:|---:|---:|
| A1 | Burgers | 1000/200 | 1/1 | 8 | 1e-3 |
| A2 | NS ν=1e-3 | 1000/200 | 10/40 | 8 | 5e-4 |
| A3 | NS ν=1e-4 | 1000/200 | 10/40 | 8 | 5e-4 |
| A4 | Shallow-water | 900/100 | 10/40 | 4 | 5e-5 |

All runs use `O=32`, `modes=16`, seed 42, and 500 epochs after a one-epoch
smoke test. Existing loaders from `pkno_v1_loaders.py` are reused so HDF5/MAT/
PDEBench layout handling remains identical to PKNO_v1.

所有实验固定 `O=32`、`modes=16`、seed 42；先做 1 epoch smoke，再做 500 epoch
full run。数据读取复用 `pkno_v1_loaders.py`，保持 HDF5、MAT 和 PDEBench 的
布局处理一致，避免 shallow-water 读取协议漂移。

## Evaluation / 评估

The trainer records train MSE, validation RL2 when a tuning split is selected,
test full-rollout RL2, per-step MSE, and spectral/gradient summaries. Formal
comparison uses the final full rollout RL2 on the fixed test indices. No test
set is used to select hyperparameters; tuning runs must use `--split-mode tuning`.
For T=40 the curriculum moves through 1/5/10/20/30/40 steps at epochs
0/40/80/120/160/200; `fallback_batches` records a long-rollout batch retried at
a shorter horizon.

正式比较使用固定测试索引上的最终完整 rollout RL2。不会使用测试集选择超参数；
调参实验必须使用 `--split-mode tuning`。T=40 的课程在 epoch
0/40/80/120/160/200 分别使用 1/5/10/20/30/40 步；`fallback_batches` 记录
发生短 horizon 回退的长 rollout batch。

## Fifth experiment / 第五实验

A joint NS run over ν=1e-3 and ν=1e-4 is reserved as a later experiment. It is
not part of the four-task promotion gate and will only be added after A1–A4 are
complete. Its purpose is to test whether one parameterized Koopman family can
interpolate across physical conditions, which single-condition runs cannot
demonstrate.

联合 NS（ν=1e-3 与 ν=1e-4）保留为第五实验，不纳入当前四项晋级门槛；它用于
验证一个参数化 Koopman family 是否能跨物理条件插值，这是单条件实验无法证明的。
