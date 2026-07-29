# Stage 0 KNO 完整实验评估

更新时间：2026-07-29

本报告补充此前的部分日志评估，集中记录已经完成 500 epoch 的 KNO
实验。失败和被替代的运行不会被删除，其详细诊断仍见
`stage0_partial_log_evaluation_2026_07_28.md`。论文数值统一取最终输出，
不按测试集最优 epoch 选择模型。

## 1. 运行与来源

| Dataset | Run | Horizon | Seed | 状态 |
|---|---|---:|---:|---|
| Burgers | `kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42` | 1 | 42 | 完成，可用 |
| NS, `nu=1e-3` | `kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_lr001_seed42_rerun1` | 40 | 42 | 完成，可用的稳定复跑 |
| NS, `nu=1e-4` | `kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_lr001_seed42_rerun1` | 40 | 42 | 完成，但为有效负结果 |
| Shallow water | `kno_koopmanlab_shallow_water_o32_m16_r8_t40_ep500_seed42` | 40 | 42 | 完成，可用 |

结构化结果分别位于：

```text
outputs/stage0_kno_baseline/kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42/metrics.csv
outputs/stage0_kno_baseline/kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_lr001_seed42_rerun1/metrics.csv
outputs/stage0_kno_baseline/kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_lr001_seed42_rerun1/metrics.csv
outputs/stage0_kno_baseline/kno_koopmanlab_shallow_water_o32_m16_r8_t40_ep500_seed42/metrics.csv
```

## 2. 统一数值摘要

Stage 0 wrapper 的 `rollout_mse_mean` 已经是各预测步 MSE 的平均值，
因此下表不再除以预测长度。

| Dataset | Mean-step MSE | Params | Mean epoch time | 科学解释 |
|---|---:|---:|---:|---|
| Burgers | `2.873726e-5` | 33,921 | 0.179 s | 有效 baseline |
| NS, `nu=1e-3` | `2.369575e-4` | 526,026 | 57.46 s | 有效 baseline |
| NS, `nu=1e-4` | `6.767658e-1` | 526,026 | 56.49 s | 完成但误差过高 |
| Shallow water | `6.913336e-5` | 526,026 | 101.29 s | 有效 baseline |

上述 epoch time 包含每轮训练及随后进行的测试集评估，不应写成纯训练
吞吐量。

## 3. 分数据集分析

### 3.1 Burgers

Burgers 运行完整结束，测试 MSE 从 `1.846e-2` 稳定下降至
`2.874e-5`，最终 epoch 同时是该日志中的最低测试 MSE。该结果说明固定、
mode-indexed Koopman 矩阵在单步 Burgers 映射上可以形成强且计算成本低的
baseline。由于 wrapper 未导出与后续 PyTorch 实现一致的 relative L2，
当前只能使用 MSE 比较，不能据此声称 KNO 在所有指标上优于其他方法。

### 3.2 Navier--Stokes, `nu=1e-3`

原始 `lr=0.005` 运行在约 epoch 27 发散，不能使用。降低到
`lr=0.001` 的稳定复跑完整训练 500 epoch，最终 mean-step MSE 为
`2.370e-4`。该复跑证明 KNO 在当前 10-step 输入、40-step输出协议下可以
稳定处理 `nu=1e-3`，但超参数已不同于最初失败运行，论文和附录必须保留
这一稳定性调整。

### 3.3 Navier--Stokes, `nu=1e-4`

降低学习率后的运行没有数值发散并完成 500 epoch，但最终 mean-step MSE
仍达到 `0.6768`。这是“执行成功、科学结果失败”的典型记录：它可以进入
匹配协议的结果表，但必须作为低粘度 40-step rollout 的困难或负结果，
不能包装为长期稳定性成功。

### 3.4 Shallow water

完整复跑的最终 mean-step MSE 为 `6.913e-5`。此前被服务器中断的部分
运行只用于说明优化过程未明显发散，现已由该完整结果替代。该 KNO 结果是
该结果可作为 shallow-water 的 KNO MSE baseline；relative L2 仍缺失。

## 4. 可用于论文的结论

- KNO 在 Burgers、NS `nu=1e-3` 和 shallow-water 上形成了可用的最终
  epoch MSE baseline。
- KNO 在 NS `nu=1e-4` 的 40-step 任务上没有达到可靠精度，低粘度长
  rollout 仍是未解决区间。
- Burgers 使用 33,921 个参数；二维 temporal tasks 使用 526,026 个参数。
- 当前结论仅来自 seed 42，不能报告 mean/std 或统计显著性。

## 5. 仍需补充

1. 在 KNO wrapper 中补充与 Stage 1/3/4 完全一致的 step/full relative L2。
2. 使用完整测试集计算逐步 relative-L2 曲线和频谱指标。
3. 增加统一 seeds，并将测试评估从逐 epoch 模型选择流程中移出。
4. 保留 `lr=0.005` 发散日志和 shallow-water 中断日志，不覆盖失败证据。
