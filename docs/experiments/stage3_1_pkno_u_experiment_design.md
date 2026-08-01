# Stage3_1 PKNO-U 实验设计

## 1. 实验目标

Stage3_1 只回答两个预注册问题：

```text
RQ1: 高通 latent U-Net 能否降低高频误差，同时不损害长 rollout 稳定性？
RQ2: 物理条件主导的 K_k 是否比状态摘要直接条件化更稳定？
```

第一批实验不改变训练损失，不引入 RL2 loss，也不同时更改频率生成策略。这样结果可归因于
PKNO-U 的 U-Fourier 支路和条件设计。

## 2. 公平比较

所有比较在相同的数据切分、`t_in`、`t_out`、seed、MSE 训练目标、训练 epoch 和 evaluator
下进行。对每个数据集至少比较：

| ID | 模型 | 条件模式 | U-Net |
|---|---|---|---|
| P0 | Stage3_0 Param-KNO | 现有 | 无 |
| U0 | PKNO-U | A `physical_only` | 有 |
| U1 | PKNO-U | B `physical_compact_state` | 有 |
| U2 | PKNO-U | C `physical_gated_state` | 有 |

在 U0 已通过稳定性门槛后，可加 U0-no-U-Net 作为结构消融；该运行仅关闭 U-Net 支路，
其余配置不变。

## 3. 主指标与停止条件

主排名指标：

```text
full_rollout_relative_l2
```

必须同时报告 step RL2、prediction MSE、逐步 MSE、高/中/低频 spectral RL2、gradient RL2、
每 epoch 时间和参数量。`spectral_metrics.csv` 目前仅覆盖首个 evaluation batch，只作为诊断，
不能单独支持论文结论。

下列任意一项发生时，full run 不得继续扩大配置：

```text
loss 或 prediction 出现 NaN/Inf
stability_diagnostics.csv 的 matrix_spectral_max 超过 max_operator_norm + 数值容差
latent_rms 持续单调暴涨
GPU peak memory 已无安全余量
```

## 4. 执行顺序

```text
1. 单元测试和 CPU forward/backward smoke
2. Burgers 1 epoch GPU smoke: U0 -> U1 -> U2
3. NS v1e-3 1 epoch GPU smoke: U0 -> U1 -> U2
4. NS v1e-4 1 epoch GPU smoke: U0 -> U1 -> U2
5. Shallow-water 1 epoch GPU smoke: U0 only, decompose=2
6. 先完成 U0 full，再依据稳定诊断开启 U1/U2 full
```

Shallow-water 不得直接把 `decompose` 提升至 8。只有 `decompose=2` 完整稳定后，才做
`2 -> 4` 的单变量扫描；每次只改变一个结构或数值参数。

## 5. 参数化泛化限制

当前单独的 NS `nu=1e-3`/`nu=1e-4` run 中，粘度对每个训练 run 是常数。它们可用于比较
稳定性和困难度，不能证明 `K_k(c_phys)` 对未见粘度泛化。该结论需要后续联合训练至少两个
粘度，并在未见粘度上测试。

## 6. 结果记录

每个真实启动的 run 先追加到 `results/experiment_result_inventory.csv`。成功、失败、协议不匹配
均保留，禁止用成功复跑覆盖失败条目。阶段报告只在获得匹配协议结果后新建。
