# Stage3_0 PKNO 实验设计

本阶段目标是绕过 Stage 1/2，先验证 PKNN 的参数化 Koopman 思想能否融入 KNO。当前代号：

```text
stage3_0_param_kno
```

## 1. 核心问题

Stage 0 已经证明四个 KNO baseline 能跑通：

- `nsv1e3`
- `nsv1e4`
- `burgers`
- `shallow_water`

Stage3_0 不重新证明 KNO 能跑，而是回答：

```text
共享 observable 空间 + 条件化 K_k 是否比 fixed KNO 更适合跨物理条件和长 rollout？
```

## 2. 从 PKNN 到 KNO 的迁移

PKNN 参考实现的核心思想是：

```text
Psi_theta(x)              # shared dictionary, independent of parameter
K_phi(u)                  # parameter-conditioned Koopman matrix
Psi_theta(y) ~= K_phi(u) Psi_theta(x)
```

KNO 的核心结构是：

```text
z_t = Psi_theta(u_t)
z_hat_t = FFT(z_t)
z_hat_{t+1,k} = K_k z_hat_{t,k}
u_{t+1} = D_theta(IFFT(z_hat_{t+1}))
```

Stage3_0 合并后采用：

```text
z_t = Psi_theta(history)
c_n_state = S_theta(history)
K_k = K0_k + DeltaK_phi(freq(k), c_static, c_n_state)
z_hat_{t+1,k} = K_k z_hat_{t,k}
u_pred = D_theta(IFFT(z_hat_{t+1}))
```

这里最重要的设计边界是：

```text
Psi_theta 不接收 c_static 或 c_n_state
K_k 接收 c_static 和 c_n_state
```

这样才能保持 PKNN 的 shared dictionary / common observable space 思想。如果把 `c` 直接拼进 `Psi_theta`，模型会退化成普通条件 encoder，不再清楚验证参数化 Koopman family 的贡献。

更详细的条件来源、`c` 与 `c_n` 区分、状态摘要和共享字典设计见：

```text
docs/models/stage3_0_condition_and_dictionary_design.md
```

## 3. 四个数据集的条件设计

| Dataset | Script | Explicit condition `c_static` | Dynamic condition `c_n_state` | 目的 |
|---|---|---|---|---|
| Burgers | `train_pkno_burgers.py` | `log10_reynolds, dx, sub, is_periodic` | mean/std/RMS, gradient, boundary, spectral energy ratios | 快速验证 1D 频域参数化 Koopman 是否可训练 |
| NS v1e-3 | `train_pkno_ns_v1e3.py` | `log10(viscosity), dx, dy, dt, t_in, t_out, sub` | mean/std/RMS, gradient, boundary, spectral energy ratios | 长 rollout 与低粘性条件 |
| NS v1e-4 | `train_pkno_ns_v1e4.py` | 同上，`viscosity=1e-4` | 同上 | 更难 NS 条件，与 v1e-3 形成条件差异 |
| Shallow-water | `train_pkno_shallow_water.py` | `dx, dy, dt, t_in, t_out, sub, radial_dam_break_flag` | 水深历史窗口摘要 | 用边界/初态强影响的 2D PDE 检验条件化 |

说明：

- 对单个数据文件内物理参数不变的数据，`c_static` 主要提供全局物理/网格条件，真正的样本级动态条件来自 `c_n_state`。
- NS v1e-3/v1e-4 是最接近参数化 Koopman 的当前数据组合，因为 viscosity 明确不同。
- 若后续能获得 `nu=5e-4` 或可生成中间 viscosity 数据，应优先做参数插值实验。

## 4. 当前实现文件

```text
src/pkno/dictionaries/shared_dictionary.py
src/pkno/operators/koopman_parameterized.py
src/pkno/models/param_kno.py
src/pkno/trainers/train_rollout.py
src/pkno/data/stage3_loaders.py

experiments/stage3_0/train_pkno_burgers.py
experiments/stage3_0/train_pkno_ns_v1e3.py
experiments/stage3_0/train_pkno_ns_v1e4.py
experiments/stage3_0/train_pkno_shallow_water.py
```

## 5. 输出

每个 run 写入：

```text
outputs/stage3_0_param_kno/<run_name>/
  args.json
  config.yaml
  env.txt
  metrics.csv
  rollout_error_by_step.csv
  spectral_metrics.csv
  checkpoint_best.pt      # optional, git ignored
  checkpoint_last.pt      # optional, git ignored
```

主指标：

- `test_full_rel_l2`
- `test_step_rel_l2`
- `high_band_spectral_rel_l2`
- `gradient_rel_l2`
- `rollout_error_by_step.csv`

## 6. 推荐对照

Stage3_0 的第一轮结论不应只看是否超过所有 baseline，而应看趋势：

| 对照 | 判断 |
|---|---|
| Stage0 KNO vs Stage3_0 Param-KNO | 参数化 K 是否降低 full rollout 或高频误差 |
| NS v1e-3 vs NS v1e-4 | viscosity 条件是否被模型稳定吸收 |
| Burgers smoke/full | 1D operator shape 和训练稳定性 |
| Shallow-water | 条件化是否帮助边界/初态主导的 2D rollout |

更典型的后续 PKNO 实验：

```text
train on nu = {1e-3, 1e-4}
test on unseen nu = 5e-4
```

如果没有中间 viscosity 数据，也可以先做联合训练：

```text
fixed KNO jointly trained on v1e-3 + v1e-4
Param-KNO jointly trained on v1e-3 + v1e-4 with log10(nu)
```

这会比单文件训练更突出 PKNO 的跨条件意义。
