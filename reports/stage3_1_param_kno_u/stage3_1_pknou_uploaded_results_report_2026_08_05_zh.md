# Stage3_1 PKNO-U 上传实验结果报告（中文）

日期：2026-08-05

## 1. 范围、来源与判定口径

本报告分析 `Latest experimental results uploaded/outputs/` 中的 KNO、iKNO、
AM-KNO、PKNO、AM-PKNO 和 Stage3_1 PKNO-U A/B/C。除特别说明外，主指标为
最终完整 rollout 的相对 L2（RL2，越低越好）。PyTorch 路线取各 `metrics.csv`
的最后一行（`epoch=499`，即训练 500 epoch），KNO 取
`evaluation_summary.json`。全部完成的正式实验均为 seed 42。

| 变体 | `condition_mode` | 设计目的 |
|---|---|---|
| PKNO-U A | `physical_only` | 主模型：Koopman 算子只显式依赖物理条件。 |
| PKNO-U B | `physical_compact_state` | 消融：加入紧凑的历史状态条件。 |
| PKNO-U C | `physical_gated_state` | 消融：对紧凑状态条件进行门控。 |

以下两项限制会影响因果解释：

1. 官方 KNO 的 NS `nu=1e-4` 使用 `ntrain=8000, ntest=200`；其余 PyTorch
   路线使用 `ntrain=1000, ntest=200`。两者测试片段不一致，KNO 数值只作库存
   记录，不能据此声称优于或劣于 KNO。
2. 稳定的 PKNO shallow-water 正式 run 使用 `decompose=4`，而 PKNO-U A/B
   使用 `decompose=8`。浅水提升不能单独归因于 U-Net。

PKNO-U C 的 NS `nu=1e-4`, T=40 和 shallow-water 未完成，分别只到 epoch
420 和 343，因此不填入结果表。AM-PKNO 的 NS `nu=1e-4`, T=20 也不是读取
遗漏：该输出目录只有 `args.json`、`config.yaml`、`env.txt`，`metrics.csv` 为
0 字节，对应日志只有 `nohup: ignoring input`；全仓库没有该 run 的 checkpoint、
rollout CSV 或评估摘要。配置指向了 T=20 数据集，但没有上传训练结果，故保留 `--`。

## 2. 精度：跨模型比较

完整 rollout RL2。粗体为该行中同协议的最低已完成值；`*` 为前述不匹配的 KNO
NS `nu=1e-4` 协议。

| 数据集与时域 | KNO | iKNO | AM-KNO | PKNO | AM-PKNO | PKNO-U A | PKNO-U B | PKNO-U C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Burgers, T=1 | 7.887e-3 | 9.602e-3 | 1.600e-2 | **5.163e-3** | 1.265e-2 | 1.475e-2 | 1.561e-2 | 1.471e-2 |
| NS, nu=1e-3, T=40 | 1.629e-2 | **1.061e-2** | 3.672e-2 | 1.632e-2 | 3.030e-2 | 5.314e-2 | 4.407e-2 | 4.961e-2 |
| NS, nu=1e-4, T=20 | 1.798e-1* | **1.597e-1** | 2.493e-1 | 1.694e-1 | -- | 2.749e-1 | 2.679e-1 | 2.759e-1 |
| NS, nu=1e-4, T=40 | 4.672e-1* | 4.776e-1 | 5.654e-1 | **4.641e-1** | 5.313e-1 | 5.639e-1 | 5.850e-1 | 未完成 |
| Shallow water, T=40 | 7.970e-3 | **4.725e-3** | 2.876e-2 | 1.486e-2 | 2.101e-2 | 1.136e-2 | 1.284e-2 | 未完成 |

### PKNO-U 直接消融

百分比按 RL2 计算，负数表示误差下降。

| 数据集与时域 | PKNO | A 相对 PKNO | B 相对 A | C 相对 A | 结论 |
|---|---:|---:|---:|---:|---|
| Burgers, T=1 | 5.163e-3 | 1.475e-2 (+185.7%) | +5.8% | -0.2% | A/B/C 均不具备竞争力。 |
| NS, nu=1e-3, T=40 | 1.632e-2 | 5.314e-2 (+225.7%) | -17.1% | -6.6% | 紧凑状态帮助 A，但仍显著落后 PKNO。 |
| NS, nu=1e-4, T=20 | 1.694e-1 | 2.749e-1 (+62.3%) | -2.6% | +0.4% | B 仅有很小收益。 |
| NS, nu=1e-4, T=40 | 4.641e-1 | 5.639e-1 (+21.5%) | +3.7% | 未完成 | A 优于 B，但均不如 PKNO。 |
| Shallow water, T=40 | 1.486e-2 | 1.136e-2 (-23.6%) | +13.1% | 未完成 | A 有改善，但分解层不匹配。 |

## 3. 模型规模与结构差异

参数量取各 run 的 `env.txt` 或最终 `metrics.csv`；2D 参数量对应所有 NS 和
shallow-water 任务。

| 方法 | 参数量：Burgers / 2D | 分解层 | modes | 条件或自适应方式 | 主要机制 |
|---|---:|---:|---|---|---|
| KNO | 0.034M / 0.526M | 8 | 16 | 无 | 固定 mode-indexed Koopman 算子。 |
| iKNO | 0.117M / 0.608M | 4 | 16 | 无 | iKNO，`p=2`。 |
| AM-KNO | 0.286M / 0.572M | 8 | all-frequency | 频率自适应 | 频率生成算子。 |
| PKNO | 0.411M / 0.916M | 8；SWE 为 4 | 16 | 物理参数化 | shared dictionary 与参数化算子。 |
| AM-PKNO | 0.382M / 0.697M | 8；SWE 为 4 | all-frequency | 自适应参数化 | AM-PKNO 路线。 |
| PKNO-U A | 0.469M / 1.136M | 8 | 16 | 仅物理条件 | 算子范数上界 0.98；后 4 层加入 latent U-Net。 |
| PKNO-U B | 0.474M / 1.148M | 8 | 16 | 物理 + 紧凑状态 | 与 A 相同的 U-Net/稳定路径。 |
| PKNO-U C | 0.474M / 1.148M | 8 | 16 | 门控物理 + 状态 | 与 A 相同的 U-Net/稳定路径。 |

相对 PKNO，A 的参数量只增加 14.2%（Burgers）和 24.0%（2D），B/C 只比 A
多约 1%。远大于此的耗时增加主要来自 U-Net 特征图计算，而非参数存储。

## 4. 训练配置与可比性

所有已记录现代 PyTorch 路线训练 500 epoch，StepLR 的 `step_size=100`、
`gamma=0.5`，`weight_decay=1e-4`。PKNO 与 PKNO-U 均采用 prediction MSE
与 reconstruction MSE 的组合；PKNO-U 记录的目标为
`5 * prediction_MSE + 0.5 * reconstruction_MSE`。RL2 是评价指标，不是训练目标。

| 数据集与时域 | Batch | PKNO LR | PKNO-U LR | 梯度裁剪：PKNO / U | 分解层：PKNO / U | 可比性 |
|---|---:|---:|---:|---:|---:|---|
| Burgers, T=1 | 64 | 1e-3 | 1e-3 | 无 / 无 | 8 / 8 | 严格的结构比较。 |
| NS, nu=1e-3, T=40 | 10 | 5e-4 | 5e-4 | 1.0 / 1.0 | 8 / 8 | 严格的结构比较。 |
| NS, nu=1e-4, T=20 | 10 | 5e-4 | 3e-4 | 1.0 / 1.0 | 8 / 8 | U-Net 与学习率混杂。 |
| NS, nu=1e-4, T=40 | 10 | 5e-4 | 3e-4 | 1.0 / 1.0 | 8 / 8 | U-Net 与学习率混杂。 |
| Shallow water, T=40 | 5 | 5e-5 | 5e-5 | 0.1 / 0.1 | 4 / 8 | U-Net 与分解层混杂。 |

当前 A/B/C 与 Stage3 PKNO 也不是严格的单变量条件消融：Stage3 PKNO 的
`state_embed_dim` 为 64，而 A/B/C 的 state embedding 为 16。后续比较必须固定
dictionary、operator generator、state width、优化器和训练协议。

## 5. 计算性能

`seconds` 为每 epoch 实测墙钟时间；总时间为 `seconds * 500 / 3600`。日志均为
NVIDIA RTX A6000、相同 Python/PyTorch/CUDA 版本，但 GPU 编号不同，服务器负载会
影响小差异。

| 数据集与时域 | AM-KNO | AM-PKNO | iKNO | PKNO | PKNO-U A | PKNO-U B | PKNO-U C |
|---|---:|---:|---:|---:|---:|---:|---:|
| Burgers, T=1 | 0.025 h | 0.043 h | 0.042 h | 0.031 h | 0.091 h | 0.084 h | 0.089 h |
| NS, nu=1e-3, T=40 | 9.42 h | 28.13 h | 9.48 h | 12.94 h | 26.74 h | 31.01 h | 27.87 h |
| NS, nu=1e-4, T=20 | 7.59 h | -- | 5.05 h | 7.48 h | 13.62 h | 14.75 h | 15.37 h |
| NS, nu=1e-4, T=40 | 8.37 h | 28.15 h | 9.58 h | 11.36 h | 29.11 h | 32.04 h | 未完成 |
| Shallow water, T=40 | 25.91 h | 65.37 h | 30.66 h | 15.18 h | 53.89 h | 55.29 h | 未完成 |

| 数据集与时域 | PKNO 每 epoch | PKNO-U A 每 epoch | A / PKNO |
|---|---:|---:|---:|
| Burgers, T=1 | 0.222 s | 0.659 s | 2.96x |
| NS, nu=1e-3, T=40 | 93.2 s | 192.6 s | 2.07x |
| NS, nu=1e-4, T=20 | 53.9 s | 98.1 s | 1.82x |
| NS, nu=1e-4, T=40 | 81.8 s | 209.6 s | 2.56x |
| Shallow water, T=40 | 109.3 s | 388.0 s | 3.55x |

只有官方 KNO evaluator 导出了推理和峰值显存，不能与未记录相同指标的其他路线
组成效率排名。

| KNO 任务 | 峰值显存 | 单步推理 | 完整 rollout 推理 |
|---|---:|---:|---:|
| Burgers, T=1 | 0.032 GB | 2.108 ms | 2.108 ms |
| NS, nu=1e-3, T=40 | 0.152 GB | 3.090 ms | 123.602 ms |
| NS, nu=1e-4, T=20 | 0.110 GB | 2.752 ms | 55.037 ms |
| NS, nu=1e-4, T=40 | 0.152 GB | 3.076 ms | 123.050 ms |
| Shallow water, T=40 | 0.274 GB | 3.667 ms | 146.677 ms |

PKNO、iKNO、AM-KNO、AM-PKNO、PKNO-U 没有记录可比较的峰值显存、单步推理
时间和完整 rollout 推理时间。`checkpoint_unet=true` 只说明 PKNO-U 使用了激活
检查点，不能替代实测显存。

## 6. A/B/C 消融解释与诊断

### A：仅物理条件

A 保留了最清晰的物理解释：同一次 rollout 内算子只依赖显式物理条件。但在严格匹配
的 Burgers 与 NS `nu=1e-3`, T=40 上，A 分别比 PKNO 高 185.7% 和 225.7%。稳定
算子约束避免了算子数值爆炸，却没有恢复 PKNO 的预测精度。

浅水是唯一正向信号：A 的 RL2 从 `1.486e-2` 降至 `1.136e-2`，预测 MSE 从
`9.587e-3` 降至 `5.606e-3`，梯度 RL2 从 1.017 降至 0.517。但该结果受 r=4/r=8
混杂，仍只是待验证的线索。

### B：紧凑状态条件

B 在 NS `nu=1e-3`, T=40 相对 A 改善 17.1%，在 NS `nu=1e-4`, T=20 改善 2.6%，
说明低维状态摘要偶尔有用；但 Burgers、NS T=40、shallow-water 均更差，并且它是
已完成长 NS run 中最慢的 PKNO-U 变体。

B 每一步都用滚动预测历史重新生成条件，使算子不再是固定物理算子，状态估计误差可
直接进入下一步 transition generator。当前混合表现与这一长期传播代价一致。

### C：门控状态条件

C 在已完成的 Burgers 上相对 A 改善 0.2%，在 NS `nu=1e-3`, T=40 改善 6.6%，在
NS `nu=1e-4`, T=20 恶化 0.4%。两个最重要的长时域任务尚未完成，因此还不能作
门控设计结论。已完成 C run 的 `condition_gate` 均值约为 0.5；这不能证明门坏了，
但也没有显示它学习到了有意义的时间/状态选择，应先检查按样本和 rollout step 的门值
分布。

### 稳定性与高频诊断

已完成 PKNO-U 的最大记录算子谱范数在 0.877--0.977，低于配置上界 0.98，latent RMS
也保持有界。因此问题不是可见的算子发散，而是稳定但不够准确。

相对 PKNO，A 的梯度 RL2 在 Burgers（0.1046 vs. 0.0268）、NS `nu=1e-3`
（0.1292 vs. 0.0463）、NS `nu=1e-4`, T=20（0.7865 vs. 0.6142）和 T=40
（0.9583 vs. 0.9150）均更差；仅 shallow-water 改善（0.5175 vs. 1.0174）。

`spectral_metrics.csv` 只覆盖第一个 evaluation batch，高频真实能量很小时高频
relative error 会病态放大，不能进入论文主结论。更重要的是所有已完成 run 的
`unet_highpass_rms` 都记录为 0；这不能单独证明 U-Net 残差未工作，但当前日志无法
证明其提供了非零高频补偿，必须优先检查。

## 7. 后续实验建议

### P0：先验证 U-Net 路径

1. 对每个 U-Net layer 记录 raw output RMS、high-pass RMS、实际加到 latent 的缩放
   residual RMS，并在全测试集聚合。
2. 记录各 U-Net 的梯度范数和参数更新范数；若高通或梯度路径近零，当前消融无效。
3. 记录 `torch.cuda.max_memory_allocated`、`max_memory_reserved`、单步推理时间和
   T=40 完整 rollout 推理时间。

### P1：使 U-Net 结论具备因果性

1. 在完全相同的 PKNO-U 架构中，运行 `hf_residual_scale=0`（或去除 U-Net）的控制组；
   固定稳定算子、dictionary、state width、optimizer、seed、decompose。该实验才能
   隔离 U-Net。
2. 在 shallow-water 同时运行 PKNO 与 PKNO-U 的 r=4 和 r=8，形成 2x2 对照，分离
   U-Net 与分解层深度的影响。
3. NS `nu=1e-4` 的 A 需用 PKNO 的 `lr=5e-4` 复跑，之后才可归因于架构。
4. P0 通过后再完成 C 的 T=40/shallow-water；否则继续消耗完整 run 的价值较低。

### P2：让参数化真正学习物理族

当前每个数据文件内物理条件向量对样本是常量，单文件训练并不能真正检验 `c_n` 是否
学习了一族物理算子。至少应混合 `nu=1e-3` 与 `nu=1e-4` 联合训练，并做粘度插值和
外推测试。

长时稳定性方面，建议保持算子生成器只依赖显式物理向量；若需要状态信息，把紧凑状态
用于 decoder correction 或初始 latent adaptation，而非每步直接喂给 Koopman 矩阵。若
保留动态状态条件算子，应约束其增量并对 state code 做时间滤波，避免预测误差反过来
改变 transition。

### P3：目标函数与评估协议

继续以 MSE 作为主训练目标，不建议改为 RL2 loss。可以做受控的辅助项消融：late-step
MSE 加权、final-step MSE、gradient/spectral loss。最终至少报告 full-rollout RL2、逐步
误差、最终步误差、全测试集谱指标、运行时间、显存，以及三个以上 seed 的均值和标准差。

## 8. 决策结论

现有证据不足以用 PKNO-U 替代 PKNO 作为默认模型。A 只有一个仍受混杂的浅水正向信号；
B/C 没有稳定的状态条件改进；且所有变体均显著增加计算成本。当前最有价值的路线是先
验证高通 U-Net 路径，再完成严格的 no-U-Net/decompose 对照，最后在真正多物理参数的
联合训练集上判断参数化设计。
