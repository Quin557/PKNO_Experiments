# Stage 1 AM-KNO 完整实验评估

更新时间：2026-07-29

Stage 1 使用 all-frequency、frequency-to-matrix generator 替换 KNO 的独立
mode-indexed 矩阵，但不使用当前物理条件或状态摘要对 Koopman 矩阵进行
动态调制。本报告分析当前已经完成的三个 500-epoch 实验，并与上一阶段
KNO 对比。

## 1. 运行与来源

| Dataset | Run | Horizon | Initial LR | 状态 |
|---|---|---:|---:|---|
| Burgers | `amkno_burgers_o32_allfreq_r8_ep500_seed42` | 1 | `1e-3` | 完成，可用 |
| NS, `nu=1e-3` | `amkno_ns_v1e3_o32_allfreq_fact1_r8_t40_ep500_seed42` | 40 | `5e-4` | 完成，可用 |
| NS, `nu=1e-4` | `amkno_ns_v1e4_o32_allfreq_fact1_r8_t40_ep500_seed42` | 40 | `3e-4` | 完成，但为有效负结果 |

每个运行的 `args.json`、`config.yaml`、`env.txt` 和逐 epoch
`metrics.csv` 均保存在对应 `outputs/stage1_0_am_kno/<run_name>/` 目录。

## 2. 指标口径

Stage 1 的 `test_pred_mse` 是所有预测步 MSE 之和。论文使用：

```text
mean_step_mse = final_test_pred_mse / t_out
```

所有正式数值取 epoch 499。下面的 best epoch 只用于观察训练波动，不用于
测试集选模。

## 3. 最终结果

| Dataset | Mean-step MSE | Step Rel. L2 | Full Rel. L2 | Params | Epoch time |
|---|---:|---:|---:|---:|---:|
| Burgers | `1.266553e-4` | `1.599594e-2` | `1.599594e-2` | 286,242 | 0.177 s |
| NS, `nu=1e-3` | `1.198112e-3` | `3.503399e-2` | `3.672308e-2` | 571,883 | 68.98 s |
| NS, `nu=1e-4` | `9.540320e-1` | `5.167857e-1` | `5.654106e-1` | 571,883 | 66.14 s |

Epoch time 是去除首轮初始化后 499 个 epoch 的平均值，并包含每轮测试评估。

## 4. 收敛与稳定性

| Dataset | Initial mean-step MSE | Final mean-step MSE | 降幅 | Diagnostic best epoch |
|---|---:|---:|---:|---:|
| Burgers | `5.530e-2` | `1.267e-4` | 99.77% | 463 |
| NS, `nu=1e-3` | `3.068e-1` | `1.198e-3` | 99.61% | 495 |
| NS, `nu=1e-4` | `2.251` | `0.954` | 57.63% | 397 |

Burgers 与 NS `nu=1e-3` 的训练过程明显收敛，最终值与最低值接近。
低粘度 NS 虽然没有数值发散，但在约 epoch 397 后只在高误差区间波动；
完成训练不等于解决了长期 rollout。

## 5. 相对 Stage 0 KNO 的解释

- Burgers AM-KNO 的 MSE 是 KNO 的约 4.41 倍。
- NS `nu=1e-3` AM-KNO 的 mean-step MSE 是 KNO 的约 5.06 倍。
- NS `nu=1e-4` AM-KNO 的 mean-step MSE 是 KNO 的约 1.41 倍，并且
  full-rollout relative L2 达到 `0.565`。

因此，当前证据不支持“仅使用频率生成器即可优于固定 KNO”。它更适合作为
结构对照：all-frequency matrix generation 可以稳定训练，但缺少条件依赖
传播时，三个已有任务的 MSE 都未超过 KNO。

## 6. 可用于论文的结论与限制

- AM-KNO 是有效的已完成 baseline，不是失败运行。
- NS `nu=1e-4` 是有效负结果，必须保留在匹配的 40-step 对比中。
- 三个运行都只有 seed 42，不能报告方差或显著性。
- `spectral_metrics.csv` 当前只覆盖首个 evaluation batch，不用于论文结论。
- NS `nu=1e-4` 的 `args.json` 中 `output_dir` 保留了旧的 Stage 0 字段，
  但 `stage`、`run_name` 和实际结果目录均指向 Stage 1；后续生成配置时应修正
  该元数据，避免自动汇总误判。

## 7. 下一步

1. 补跑 shallow-water AM-KNO，完成四数据集矩阵。
2. 在完整测试集上重新计算逐步 relative L2 和频谱指标。
3. 增加统一 seeds，并采用 validation checkpoint 后一次性测试的流程。
