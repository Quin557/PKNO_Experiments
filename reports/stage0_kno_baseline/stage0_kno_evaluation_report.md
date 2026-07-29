# Stage 0 KNO Evaluation Report

更新时间：2026-07-29

## 1. 范围

仅覆盖 Stage 0 KNO baseline：

1. Burgers
2. Navier-Stokes `nu=1e-3`
3. Navier-Stokes `nu=1e-4`
4. Shallow Water

不讨论 AM-KNO、Param-KNO 或 Stage 4。

## 2. checkpoint 检查结果

已扫描：

```text
outputs/stage0_kno_baseline
```

checkpoint 清点文件：

```text
reports/stage0_kno_baseline/checkpoint_inventory.md
```

结论：

- 当前没有任何可加载的 `checkpoint_last.pt`
- `time_error.pt`、`metrics.csv`、预测张量不算 checkpoint
- 因此 evaluation-only smoke test 目前不能真正启动

## 3. 成功 / 失败 / 不完整

### 成功

- `kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42`
- `kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_lr001_seed42_rerun1`
- `kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_lr001_seed42_rerun1`
- `kno_koopmanlab_shallow_water_o32_m16_r8_t40_ep500_seed42`

上述“成功”仅表示已有完整结构化结果；它们当前仍缺少可重载 checkpoint。

### 失败 / 不完整

- 旧的 NS `lr=0.005` 长训日志：发散或中断
- 早先被 kill 的 shallow-water / NS 中断日志：保留为不完整证据
- 所有现有 Stage 0 输出：缺少可加载 checkpoint

## 4. 当前任务配置

| Run | config / args |
|---|---|
| Burgers | `outputs/stage0_kno_baseline/kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42/args.json` |
| NS v1e-3 | `outputs/stage0_kno_baseline/kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_lr001_seed42_rerun1/args.json` |
| NS v1e-4 | `outputs/stage0_kno_baseline/kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_lr001_seed42_rerun1/args.json` |
| Shallow Water | `outputs/stage0_kno_baseline/kno_koopmanlab_shallow_water_o32_m16_r8_t40_ep500_seed42/args.json` |

这些配置保持：

- 数据文件和预处理
- train/test split
- `ntrain` / `ntest`
- `t_in` / `t_out`
- resolution / subsampling
- operator size / modes / decompose
- batch size
- learning rate
- epochs = 500
- seed = 42

## 5. 评估指标定义

### 5.1 step_rel_l2

对每个样本、每个预测步计算：

```text
||prediction - target||_2 / (||target||_2 + eps)
```

再对测试样本和预测步取平均。

### 5.2 full_rollout_rel_l2

对每个样本，将完整 rollout 的时间维和空间维展平后计算 relative L2，再对完整测试集平均。

### 5.3 rollout_error_by_step

每个预测步输出：

- mse
- rel_l2
- 样本数

必须覆盖完整测试集，不能只看单个 batch。

### 5.4 rollout_growth_slope

对完整测试集的 per-step rel_l2 均值曲线做 OLS 线性拟合，输出 slope 和拟合区间。

### 5.5 complexity

- trainable parameter count
- peak GPU memory
- inference ms/step
- complete-rollout latency

GPU 计时必须使用 `torch.cuda.synchronize()`，包含 warm-up 和多次重复。

### 5.6 次优先级指标

- low / mid / high spectral-band relative L2
- gradient relative L2

频段边界会写入配置和结果文件，不事后调整。

## 6. 结果表

当前只保留已有 MSE 级结果，尚未生成新的 checkpoint-based evaluation-only 指标。

| Run | Status | Note |
|---|---|---|
| Burgers | complete | 现有结果可用，但无 checkpoint |
| NS v1e-3 | complete | 现有结果可用，但无 checkpoint |
| NS v1e-4 | complete | 现有结果可用，但无 checkpoint |
| Shallow Water | complete | 现有结果可用，但无 checkpoint |

## 7. rollout 稳定性

- Burgers：稳定
- NS v1e-3：原始 `lr=0.005` 运行发散，`lr=0.001` 复跑稳定
- NS v1e-4：当前结果可完成，但误差显著高，属于困难/负结果
- Shallow Water：当前结果趋势稳定

## 8. 可追溯路径

```text
reports/stage0_kno_baseline/checkpoint_inventory.md
reports/stage0_kno_baseline/stage0_partial_log_evaluation_2026_07_28.md
reports/stage0_kno_baseline/stage0_completed_experiment_evaluation_2026_07_29.md
outputs/stage0_kno_baseline/<run_name>/
logs/stage0_kno_baseline/<run_name>.log
```

## 9. 尚未完成

1. 为四个 Stage 0 run 补出可重载 `checkpoint_last.pt`
2. 用独立 evaluation-only 脚本重新加载 checkpoint
3. 产出完整 `step_rel_l2` / `full_rollout_rel_l2` / `rollout_growth_slope`
4. 补齐 `complexity.csv` 与 `evaluation_summary.json`
5. 再生成第二优先级的 spectral / gradient 指标

## 10. 下一步建议

先按以下顺序补 checkpoint：

```text
1. Burgers smoke
2. Burgers 500 epoch replay if needed
3. NS v1e-3
4. NS v1e-4
5. Shallow Water
```

对应的 evaluation-only 入口已经准备好：

```text
experiments/official_kno/evaluate_koopmanlab_checkpoint.py
```
