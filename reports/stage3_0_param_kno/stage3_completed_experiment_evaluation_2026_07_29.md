# Stage 3 Param-KNO 完整实验评估

更新时间：2026-07-29

Stage 3 保留 KNO 的 mode-indexed base matrices，并同时引入条件依赖修正和
shared hybrid observable dictionary。当前实验不能把这两个组件的贡献分别
归因，只能分析它们的联合效果。

## 1. 运行与来源

| Dataset | Run | Horizon | 状态 |
|---|---|---:|---|
| Burgers | `pkno_burgers_o32_m16_r8_ep500_seed42` | 1 | 完成，可用 |
| NS, `nu=1e-3` | `pkno_ns_v1e3_o32_m16_r8_t40_ep500_seed42` | 40 | 完成，可用 |
| NS, `nu=1e-4` | `pkno_ns_v1e4_o32_m16_r8_t20_ep500_seed42` | 20 | 完成，仅诊断 |
| NS, `nu=1e-4` | `pkno_ns_v1e4_o32_m16_r8_t40_ep500_seed42` | 40 | 完成，但为有效负结果 |
| Shallow water | `pkno_shallow_water_o32_m16_r8_t40_ep500_seed42_lr1e4_ds1e2` | 40 | 完成，混合结果 |

结构化输出位于 `outputs/stage3_0_param_kno/<run_name>/`。正式结果统一使用
epoch 499；20-step NS 运行不进入 40-step 主表。

## 2. 最终结果

| Dataset | Horizon | Mean-step MSE | Step Rel. L2 | Full Rel. L2 | Params | Epoch time |
|---|---:|---:|---:|---:|---:|---:|
| Burgers | 1 | `3.115199e-5` | `5.162571e-3` | `5.162571e-3` | 411,001 | 0.239 s |
| NS, `nu=1e-3` | 40 | `2.397234e-4` | `1.570295e-2` | `1.631617e-2` | 915,585 | 100.84 s |
| NS, `nu=1e-4` | 20 | `7.563229e-2` | `1.453016e-1` | `1.693865e-1` | 915,585 | 52.53 s |
| NS, `nu=1e-4` | 40 | `6.667135e-1` | `4.021632e-1` | `4.641091e-1` | 915,585 | 91.89 s |
| Shallow water | 40 | `2.396786e-4` | `1.482857e-2` | `1.486202e-2` | 915,585 | 121.89 s |

Mean-step MSE 是最终 `test_pred_mse / t_out`；epoch time 去除首轮初始化，
并包含每轮测试评估。

## 3. 收敛诊断

| Dataset | Initial MSE | Final MSE | 降幅 | MSE diagnostic best epoch |
|---|---:|---:|---:|---:|
| Burgers | `2.275e-2` | `3.115e-5` | 99.86% | 396 |
| NS, `nu=1e-3` | `9.617e-2` | `2.397e-4` | 99.75% | 498 |
| NS, `nu=1e-4`, T=20 | `5.962e-1` | `7.563e-2` | 87.31% | 485 |
| NS, `nu=1e-4`, T=40 | `2.093` | `6.667e-1` | 68.14% | 376 |
| Shallow water | `1.386e-2` | `2.397e-4` | 98.27% | 325 |

Burgers 与 NS `nu=1e-3` 收敛充分，最终 epoch 与最低测试区间接近。
Shallow-water 在中后期存在较明显波动：最低 diagnostic MSE 出现在 epoch
325，而论文仍使用最终 epoch，避免利用测试集选择更有利结果。

## 4. 分任务分析

### 4.1 Burgers

Param-KNO 的最终 relative L2 为 `5.163e-3`，比 AM-KNO 低 67.7%；
mean-step MSE 比 AM-KNO 低约 75.4%。与 KNO 相比，MSE 高约 8.4%，且
KNO 缺少匹配的 relative L2，因此只能写成“在统一 PyTorch relative-L2
口径下优于 AM-KNO 和当前 PKNO”，不能宣称全面优于 KNO。

### 4.2 Navier--Stokes, `nu=1e-3`

Param-KNO 的 full-rollout relative L2 为 `1.632e-2`，相对 AM-KNO
降低 55.6%；mean-step MSE 相对 AM-KNO 降低约 80.0%，同时只比 KNO
高约 1.2%。这是当前最有力的正向结果，说明 condition-dependent
propagation 与 hybrid dictionary 的组合明显优于纯频率生成 AM-KNO。

### 4.3 Navier--Stokes, `nu=1e-4`

40-step Param-KNO 相对 AM-KNO 将 full-rollout relative L2 从 `0.565`
降至 `0.464`，降低 17.9%；mean-step MSE 也比 KNO 低约 1.5%。然而绝对
误差仍然很高，所以该任务仍是有效负结果。

20-step 运行的 mean-step MSE 是 `0.0756`，而 40-step 上升到 `0.6667`，
约为前者 8.81 倍；full-rollout relative L2 从 `0.169` 上升到 `0.464`。
这能用于诊断 horizon-induced degradation，但不能把 T=20 与 T=40 放在
同一个主比较表中。

### 4.4 Shallow water

Param-KNO 完成了 40-step rollout，最终 full relative L2 为 `1.486e-2`，
但 mean-step MSE 是 KNO 的约 3.47 倍，因此不支持优于 KNO的结论。该运行
采用 dataset-specific 稳定设置：`lr=5e-5`、`decompose=4`、
`delta_scale=0.005` 和 `max_grad_norm=0.1`；这些配置必须在附录中披露。

## 5. 可用于论文的结论与限制

- 正向证据集中在 Burgers relative L2 和 NS `nu=1e-3` 的 40-step 结果。
- NS `nu=1e-4` 与 shallow-water 必须保留为负结果或混合结果。
- 当前比较只能解释 condition-dependent propagation 与 hybrid dictionary
  的联合贡献，不能声称完成了细粒度组件消融。
- 所有正式运行仅有 seed 42；频谱 CSV 也尚未覆盖完整测试集。
- NS `nu=1e-4`, T=40 的 `args.json` 中 `output_dir` 保留旧 Stage 0 字段，
  但实际目录、`stage` 和 `run_name` 均指向 Stage 3；后续应清理该元数据。

## 6. 下一步

1. 补统一 seeds，重点复核 NS `nu=1e-3` 的优势是否稳定。
2. 修正逐步 relative-L2 与全测试集频谱聚合，再计算 growth slope。
3. 若需要分别归因 dictionary 和 condition-dependent propagation，必须新增
   专门消融；当前 Stage 0/1/3/4 结构矩阵不能完成这一归因。
