# Stage 3 PKNO 阶段性实验评估

更新时间：2026-07-29

本阶段的 `ParamKNO` 实现对应论文中的最终模型 **PKNO**。当前论文范围
只包含 KNO、AM-KNO 和 PKNO。

本报告在前两阶段基础上评估 PKNO：先以 KNO 作为固定 mode-indexed
operator baseline，再以 AM-KNO 作为纯 frequency-generated 对照，最后
检验 condition-dependent propagation 与 shared hybrid dictionary 的联合收益。

## 1. 运行与来源

| Dataset | Run | Horizon | 论文用途 |
|---|---|---:|---|
| Burgers | `pkno_burgers_o32_m16_r8_ep500_seed42` | 1 | 主 relative-L2 对比 |
| NS, `nu=1e-3` | `pkno_ns_v1e3_o32_m16_r8_t40_ep500_seed42` | 40 | 主 relative-L2 对比 |
| NS, `nu=1e-4` | `pkno_ns_v1e4_o32_m16_r8_t40_ep500_seed42` | 40 | 主 relative-L2 对比，困难区间 |
| NS, `nu=1e-4` | `pkno_ns_v1e4_o32_m16_r8_t20_ep500_seed42` | 20 | 仅用于 horizon degradation 诊断 |
| Shallow water | `pkno_shallow_water_o32_m16_r8_t40_ep500_seed42_lr1e4_ds1e2` | 40 | 暂不进入主表，匹配对照不完整 |

结构化输出位于 `outputs/stage3_0_param_kno/<run_name>/`。正式论文数值统一
使用 epoch 499，不使用测试集最优 epoch 选模。

## 2. 论文主指标

跨数据集 MSE 受场值尺度影响，而且当前 KNO wrapper 尚未导出匹配的
relative L2。当前主表因此采用 AM-KNO 与 PKNO 在同一 PyTorch evaluator
下得到的 step/full relative L2。该选择与长期 rollout 的研究问题直接对应，
但不能被解释为 KNO 已被完整比较。

| Dataset | Method | Step Rel. L2 | Full Rel. L2 | PKNO improvement |
|---|---|---:|---:|---:|
| Burgers | AM-KNO | `1.600e-2` | `1.600e-2` | -- |
|  | **PKNO** | **`5.163e-3`** | **`5.163e-3`** | **67.7%** |
| NS, `nu=1e-3` | AM-KNO | `3.503e-2` | `3.672e-2` | -- |
|  | **PKNO** | **`1.570e-2`** | **`1.632e-2`** | **55.6%** |
| NS, `nu=1e-4` | AM-KNO | `5.168e-1` | `5.654e-1` | -- |
|  | **PKNO** | **`4.022e-1`** | **`4.641e-1`** | **17.9%** |

“PKNO improvement”按 full-rollout relative L2 相对 AM-KNO 的降幅计算。
在三个拥有匹配 relative-L2 结果的任务上，PKNO 均优于 AM-KNO。

## 3. 与 KNO 的辅助比较

KNO 当前只能使用 mean-step MSE 对照，因此下表属于内部完整性检查，不作为
relative-L2 主表的排名依据。

| Dataset | KNO MSE | AM-KNO MSE | PKNO MSE | 阶段性解释 |
|---|---:|---:|---:|---|
| Burgers | `2.874e-5` | `1.267e-4` | `3.115e-5` | PKNO 接近 KNO，并显著优于 AM-KNO |
| NS, `nu=1e-3` | `2.370e-4` | `1.198e-3` | `2.397e-4` | PKNO 与 KNO 相差约 1.2%，优于 AM-KNO |
| NS, `nu=1e-4` | `6.768e-1` | `9.540e-1` | `6.667e-1` | PKNO 比 KNO 低约 1.5%，但绝对误差仍高 |
| Shallow water | `6.913e-5` | -- | `2.397e-4` | 对照不完整，暂不进入论文主表 |

当前数据不支持“AM-KNO 优于 KNO”的结论。可以支持的三阶段叙述是：纯
frequency generation 的 AM-KNO 未能保持 KNO 的 MSE；加入条件依赖传播
和 hybrid dictionary 后，PKNO 恢复到接近或略优于 KNO 的 MSE，同时在
统一 relative-L2 口径下显著优于 AM-KNO。

## 4. 分任务结论

### 4.1 Burgers

PKNO 的 full relative L2 为 **`5.163e-3`**，相对 AM-KNO 降低
**67.7%**；mean-step MSE 相对 AM-KNO 降低 75.4%，并接近 KNO。
由于 KNO relative L2 缺失，论文应写成“PKNO 在匹配 relative-L2
evaluator 下取得最低误差”，而不是“全面超过 KNO”。

### 4.2 Navier--Stokes, `nu=1e-3`

PKNO 的 full-rollout relative L2 为 **`1.632e-2`**，相对 AM-KNO
降低 **55.6%**。其 mean-step MSE 为 `2.397e-4`，与 KNO 的
`2.370e-4` 相差约 1.2%。这是当前最有力的长时预测证据。

### 4.3 Navier--Stokes, `nu=1e-4`

PKNO 相对 AM-KNO 将 full-rollout relative L2 从 `0.565` 降至
**`0.464`**，降低 **17.9%**，MSE 也略低于 KNO。但绝对误差仍然很高，
因此只能表述为“缓解低粘度误差”，不能声称已解决长期稳定性。

20-step 诊断结果的 full relative L2 为 `0.169`，不能与 40-step 主表
混合。它只说明预测长度增加是误差恶化的重要来源。

### 4.4 Shallow water

当前缺少 AM-KNO 正式结果，KNO 又缺少 matched relative L2。PKNO 虽有
完整输出，但现阶段无法形成三模型公平比较，因此暂不进入论文主结果表。
该结果继续保存在清单和内部分析中，待匹配评估完成后再决定正文或附录位置。

## 5. 论文可用结论

- 在三个 matched relative-L2 comparison 中，PKNO 均为最优模型。
- PKNO 相对 AM-KNO 的 full-rollout relative-L2 降幅分别为 67.7%、
  55.6% 和 17.9%。
- 在 NS `nu=1e-3` 上，PKNO 同时保持了与 KNO 基本相同的 mean-step MSE。
- 结构证据支持 condition-dependent propagation 与 hybrid dictionary 的
  联合贡献；当前实验不能分别归因两个组件。

## 6. 必须保留的限制

1. 所有正式结果目前只有 seed 42，不能报告 mean/std 或显著性。
2. KNO matched relative L2 尚未导出，不能宣称 PKNO 已在主指标上超过 KNO。
3. NS `nu=1e-4` 仍是高绝对误差区间。
4. Shallow-water 三模型对照尚未完成。
5. `spectral_metrics.csv` 仅覆盖首个 evaluation batch，不能进入论文结论。
