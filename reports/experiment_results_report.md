# PKNO 实验结果分析报告

更新时间：2026-07-29

本报告整理当前本地 `outputs/` 与 `logs/` 中的正式实验结果，并作为后续新增实验的持续更新入口。机器可读清单见 `results/experiment_result_inventory.csv`。论文中的数值必须能回溯到该清单的 `source` 字段。

分阶段的训练行为与局部结论见：

- `stage0_kno_baseline/stage0_completed_experiment_evaluation_2026_07_29.md`
- `stage1_0_am_kno/stage1_completed_experiment_evaluation_2026_07_29.md`
- `stage3_0_param_kno/stage3_completed_experiment_evaluation_2026_07_29.md`
- `stage4_0_am_pkno/stage4_completed_experiment_evaluation_2026_07_29.md`

## 1. 判定规则

本报告区分三个维度，避免把“跑完了”误写成“方法有效”。

1. **执行状态**：是否完成预定 epoch，是否出现发散、非有限值、服务器中断或缺少结果文件。
2. **协议状态**：数据切分、预测长度、seed 和指标实现是否满足论文主表的比较条件。
3. **科学结论**：完成的实验可能是正向结果，也可能是有效的负结果。绝对误差很高的完整 run 仍应保留，但不能包装成模型成功。

当前论文候选值统一采用 seed 42 的第 500 个 epoch（CSV 中 `epoch=499`）或官方 wrapper 的 500-epoch 最终输出。虽然训练 CSV 每个 epoch 都记录测试误差，本报告不使用测试集最优 epoch 作为论文结果，以避免 test-set model selection。Stage 3 的 20-step Navier--Stokes 结果不进入 40-step 主表。

对于 Stage 1/3/4，训练程序的 `test_pred_mse` 是各预测步 MSE 的和。因此本报告中的 `Mean-step MSE` 按 `test_pred_mse / T_out` 计算，使其与 Stage 0 wrapper 的 `rollout_mse_mean` 口径一致。

## 2. 当前结果总览

| Dataset | Method | Horizon | Mean-step MSE | Mean-step Rel. L2 | Full-rollout Rel. L2 | Params | 判定 |
|---|---|---:|---:|---:|---:|---:|---|
| Burgers | KNO | 1 | 2.874e-5 | - | - | 33,921 | 可用，缺 relative L2 |
| Burgers | AM-KNO | 1 | 1.267e-4 | 1.600e-2 | 1.600e-2 | 286,242 | 可用 |
| Burgers | Param-KNO | 1 | 3.115e-5 | 5.163e-3 | 5.163e-3 | 411,001 | 可用，当前最强 relative L2 |
| Burgers | PKNO | 1 | 8.079e-5 | 1.265e-2 | 1.265e-2 | 382,329 | 可用，结果混合 |
| NS, nu=1e-3 | KNO | 40 | 2.370e-4 | - | - | 526,026 | 可用稳定复跑 |
| NS, nu=1e-3 | AM-KNO | 40 | 1.198e-3 | 3.503e-2 | 3.672e-2 | 571,883 | 可用 |
| NS, nu=1e-3 | Param-KNO | 40 | 2.397e-4 | 1.570e-2 | 1.632e-2 | 915,585 | 可用 |
| NS, nu=1e-4 | KNO | 40 | 6.768e-1 | - | - | 526,026 | 完成但为负结果 |
| NS, nu=1e-4 | AM-KNO | 40 | 9.540e-1 | 5.168e-1 | 5.654e-1 | 571,883 | 完成但为负结果 |
| NS, nu=1e-4 | Param-KNO | 40 | 6.667e-1 | 4.022e-1 | 4.641e-1 | 915,585 | 完成但绝对误差仍高 |
| Shallow water | KNO | 40 | 6.913e-5 | - | - | 526,026 | 可用，缺 relative L2 |
| Shallow water | Param-KNO | 40 | 2.397e-4 | 1.483e-2 | 1.486e-2 | 915,585 | 可用但未优于 KNO MSE |

Stage 4 当前只有 Burgers 完整结果。Stage 4 的三个 temporal tasks 以及 Stage 1 shallow-water 尚无正式 500-epoch 输出，因此主表保留空位。

## 3. 成功且可用于当前论文的实验

### 3.1 Burgers

Param-KNO 的最终 relative L2 为 `5.163e-3`，相对 AM-KNO 降低 67.7%，相对当前 PKNO 降低 59.2%。PKNO 相对 AM-KNO 降低 20.9%，说明频率生成与条件化结合相对纯频率生成有收益，但完整组合尚未超过保留 mode-indexed base operator 的 Param-KNO。

KNO 的 MSE `2.874e-5` 略低于 Param-KNO 的 `3.115e-5`。由于 KNO wrapper 没有输出 relative L2，不能据此宣称 Param-KNO 已全面超过 KNO；当前只能说明 Param-KNO 在新模型的统一 relative-L2 口径下最好。

### 3.2 Navier--Stokes, nu=1e-3

Param-KNO 的 full-rollout relative L2 为 `1.632e-2`，比 AM-KNO 的 `3.672e-2` 低 55.6%。其 mean-step MSE 为 `2.397e-4`，与 KNO 稳定复跑的 `2.370e-4` 相差约 1.2%。这一结果支持以下有限结论：在保持 KNO 级别 mean-step MSE 的同时，条件依赖传播与 hybrid dictionary 显著改善了纯频率生成的 AM-KNO。

该结论目前只来自 seed 42，且 KNO 缺少 relative L2，因此不能写成统计显著性结论。

## 4. 完成但属于负结果或混合结果

### 4.1 Navier--Stokes, nu=1e-4, 40 steps

三种已完成方法的绝对误差都很高。Param-KNO 的 full-rollout relative L2 `0.464` 比 AM-KNO 的 `0.565` 低 17.9%，mean-step MSE `0.667` 也比 KNO 的 `0.677` 低约 1.5%，但这些改进不足以构成可靠的长时稳定性成功。

因此该任务应在论文中被表述为当前困难/失败区间：Param-KNO 相对缓解误差，但没有解决低粘度 40-step rollout。Stage 3 的 20-step结果 `full Rel. L2=0.169` 只能用于定位 horizon-induced degradation，不能与 40-step 主表混合。

### 4.2 Shallow water

Param-KNO 的 full-rollout relative L2 为 `1.486e-2`，但其 mean-step MSE `2.397e-4` 是 KNO `6.913e-5` 的 3.47 倍。当前证据不支持 Param-KNO 在 shallow-water 上优于 KNO。该 run 还使用了数据集特定的稳定设置（最终 `lr=5e-5`），必须在附录披露。

### 4.3 Full PKNO

当前只有 Burgers 结果。PKNO 优于 AM-KNO，但弱于 Param-KNO，说明 all-frequency conditional generation 并非自动优于 mode-indexed conditional correction。Temporal Stage 4 未完成前，不能回答完整 PKNO 的长时稳定性问题。

## 5. 失败、被替代与仅诊断记录

| Run | 类型 | 现象 | 处理 |
|---|---|---|---|
| Stage 0 NS nu=1e-3, lr=0.005 | 发散 | epoch 27 发生突增，后续停留在高损失平台 | 不进论文；由 lr=0.001 完整复跑替代 |
| Stage 0 NS nu=1e-4, lr=0.005 | 不稳定且未完成 | 长期高 loss，日志止于 epoch 420 | 不进论文；保留为稳定性失败证据 |
| Stage 0 shallow-water 早期副本 | 服务器中断但趋势稳定 | 只完成部分 epoch | 不进论文；由完整 500-epoch run 替代 |
| Stage 3 NS nu=1e-4, T=20 | 协议不匹配 | run 完成且误差低于 T=40，但 horizon 不同 | 仅用于诊断 horizon degradation |
| 所有 `smoke_*` run | 冒烟测试 | 只验证 shape、显存或训练链路 | 永不进入论文结果 |

## 6. 当前指标限制

- `rollout_error_by_step.csv` 只保存逐步 MSE，尚不能计算正文定义的逐步 relative-L2 growth slope。
- `spectral_metrics.csv` 只使用第一个 evaluation batch。其 high-band 值在真值频带能量极小时会异常放大，当前不得进入论文主表。
- KNO wrapper 只输出 MSE，缺少与 Stage 1/3/4 完全一致的 step/full relative L2。
- 所有正式结果目前只有 seed 42，不能报告 mean +/- std 或统计显著性。
- 当前训练循环每个 epoch 都评估 test set。论文采用最终 epoch，后续应增加 validation split，并仅在固定 checkpoint 上运行一次 test evaluation。

## 7. 计算成本快照

当前 `seconds` 包含一个完整训练 epoch 及随后 test evaluation，因此应写作 epoch wall time，而不是纯训练时间。

| Dataset | Method | Params (M) | Mean epoch wall time (s) |
|---|---|---:|---:|
| Burgers | KNO | 0.034 | 0.179 |
| Burgers | AM-KNO | 0.286 | 0.177 |
| Burgers | Param-KNO | 0.411 | 0.239 |
| Burgers | PKNO | 0.382 | 0.247 |
| NS, nu=1e-4, T=40 | KNO | 0.526 | 56.5 |
| NS, nu=1e-4, T=40 | AM-KNO | 0.572 | 66.1 |
| NS, nu=1e-4, T=40 | Param-KNO | 0.916 | 91.9 |

在 Burgers 上，AM-KNO 的参数量约为 KNO 的 8.4 倍，但 epoch wall time相近；Param-KNO 和 PKNO 分别约为 12.1 倍和 11.3 倍参数量。NS 上 Param-KNO 相对 KNO 增加约 74% 参数和约 63% epoch wall time。Stage 4 temporal 结果缺失，当前不能评价完整 PKNO 的 long-rollout 计算代价。

## 8. 后续新增数据的更新流程

每次服务器结果同步后按以下顺序更新：

1. 在 `experiment_result_inventory.csv` 追加一行，不覆盖失败记录。
2. 核对 `args.json`、`env.txt`、`metrics.csv`、日志尾部和预测 horizon。
3. 将 run 标为 `paper_candidate`、`diagnostic_only` 或 `failed`。
4. 只使用最终 epoch 或预先规定的 validation-selected checkpoint，不使用 test-set best epoch。
5. 只有 split、horizon、seed 集合与 metric implementation 一致的结果才能进入论文同一比较块。
6. 更新本报告的结果表和分析，再同步论文表格。

下一批最优先需要补充：Stage 4 三个 temporal tasks、Stage 1 shallow-water、KNO relative L2、全测试集 spectral metrics、逐步 relative L2 和至少 3 个统一 seeds。
