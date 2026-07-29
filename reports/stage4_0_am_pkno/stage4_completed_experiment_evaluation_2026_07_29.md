# Stage 4 PKNO 已完成实验评估

更新时间：2026-07-29

Stage 4 将 shared hybrid dictionary、condition encoder 与 all-frequency
matrix generator 组合为完整 PKNO。当前只有 Burgers 完成 500 epoch；因此
本报告是已完成结果分析，不代表 Stage 4 已覆盖论文要求的 temporal tasks。

## 1. 运行与来源

```text
run: ampkno_burgers_o32_allfreq_r8_ep500_seed42
source: outputs/stage4_0_am_pkno/ampkno_burgers_o32_allfreq_r8_ep500_seed42/
epochs: 500
seed: 42
operator_size: 32
decompose: 8
mode_policy: all_fft_modes
factorized_rank: 1
use_hf_residual: false
params: 382,329
```

该目录包含 `args.json`、`config.yaml`、`env.txt`、`metrics.csv`、
`rollout_error_by_step.csv` 和 `spectral_metrics.csv`。

## 2. 最终结果

| Metric | Final epoch value |
|---|---:|
| Mean-step MSE | `8.079378e-5` |
| Step Rel. L2 | `1.264571e-2` |
| Full Rel. L2 | `1.264571e-2` |
| Reconstruction MSE | `4.808588e-6` |
| Mean epoch time | 0.247 s |

Mean epoch time 去除首轮初始化并包含每轮测试评估。Burgers 是单步任务，
因此 step relative L2 与 full-rollout relative L2 相同。

## 3. 收敛分析

测试 MSE 从 epoch 0 的 `4.303e-2` 降至最终 `8.079e-5`，降幅约
99.81%，说明模型正常训练并显著收敛。最低 diagnostic MSE 出现在 epoch
488，为 `6.577e-5`；最低 full relative L2 出现在 epoch 454，为
`1.131e-2`。最终值略高于这些最低点，反映中后期存在波动。论文继续采用
最终 epoch，不使用测试集最优点。

## 4. 与现有 Burgers 对照

| Method | Mean-step MSE | Full Rel. L2 |
|---|---:|---:|
| KNO | `2.874e-5` | -- |
| AM-KNO | `1.267e-4` | `1.600e-2` |
| Param-KNO | `3.115e-5` | `5.163e-3` |
| PKNO | `8.079e-5` | `1.265e-2` |

PKNO 相对 AM-KNO 将 MSE 降低约 36.2%，relative L2 降低约 20.9%。
这说明在 all-frequency generator 上加入条件化 shared dictionary 有收益。
但 PKNO 的 MSE 是 Param-KNO 的约 2.59 倍，relative L2 是其约 2.45 倍；
它也没有达到 KNO 的 MSE。因此，当前结果属于混合结果，而不是完整 PKNO
优于所有 baseline 的证据。

## 5. 可用于论文的结论与限制

- Full PKNO 在 Burgers 上正常完成训练，并稳定优于 AM-KNO。
- Full PKNO 当前没有超过保留 mode-indexed base matrices 的 Param-KNO。
- 不能根据单步 Burgers 推断长期 rollout 稳定性。
- temporal Stage 4、统一 seeds、完整测试集频谱指标和推理成本均缺失。
- 本运行明确设置 `use_hf_residual=false`，没有使用 Stage 2 高频残差分支。

## 6. 下一步

1. 优先补齐 NS `nu=1e-3`、NS `nu=1e-4` 和 shallow-water Stage 4。
2. 保持与 Stage 0/1/3 一致的数据 split、horizon 和 seed 集合。
3. 在 temporal runs 完成前，不在论文中回答完整 PKNO 的长期稳定性问题。
