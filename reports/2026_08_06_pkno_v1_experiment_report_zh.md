# PKNO_v1 最新实验报告

日期：2026-08-06  
结果目录：`Latest experimental results uploaded/outputs/`  
主模型：`stage3_2_pkno_v1`，seed 42，500 epochs，`O=32`，`modes=16`

## 1. 执行摘要

本轮 PKNO_v1 是独立于旧 PKNO 的新路线，保留 Fourier Koopman、共享字典和 1x1 skip convolution，不包含 U-Net。主要改动是：

1. 用物理条件优先的参数化，将物理条件嵌入作为主条件；状态嵌入通过有界 gate 调制，避免 `c_n_state` 直接生成过强的动态矩阵变化。
2. 将输出改为物理场残差预测：`u_(n+1)=u_n+Decoder(z_(n+1))`。
3. 对 NS `nu=1e-4` 加入软增长包络约束，并对 NS/shallow-water 使用 `1 -> 5 -> 10 -> 40` 的 rollout curriculum。
4. 训练目标仍是 MSE（加重构项和稳定性正则）；RL2 只用于验证、最终报告和模型比较。

结果显示，PKNO_v1 在四项主实验中有两项超过旧 PKNO：NS `nu=1e-4,T=40` 和 shallow-water；另外 NS `nu=1e-4,T=20` 也有所改善。但 Burgers 和 NS `nu=1e-3,T=40` 退化，因此没有达到设计文档规定的“四项全部超过旧 PKNO”的 promotion gate，当前不应替换论文中的 PKNO 结果。

## 2. 结果与比较协议

- 所有主结果均为完整 500 epoch、seed 42。
- Burgers：train/test = 1000/200，单步预测。
- NS `nu=1e-3`、`nu=1e-4`：PKNO、iKNO、AM-KNO、AM-PKNO、PKNO_v1 使用 1000/200；PKNO_v1 另使用 1200:1399 作为 validation，不改变 test=1000:1199。
- shallow-water：train/test = 900/100。
- 旧 PKNO shallow-water 的实际配置是 `decompose=4`，本轮 PKNO_v1 shallow-water 实际配置是 `decompose=8`，所以该项是有利于 V1 的非严格同配置比较，必须在后续用 `decompose=4` 重跑确认。
- 官方 KNO 的 NS `nu=1e-4` 使用 8000/200，与其他模型的 1000/200 不同，表中用 `*` 标注，仅作结果参考，不作严格优劣结论。

## 3. 完整 rollout RL2

数值越低越好；粗体表示该行在已完成结果中的最低值。

| 任务 | KNO | iKNO | AM-KNO | 旧 PKNO | AM-PKNO | PKNO_v1 |
|---|---:|---:|---:|---:|---:|---:|
| Burgers, T=1 | 7.887e-3 | 9.602e-3 | 1.600e-2 | **5.163e-3** | 1.265e-2 | 7.694e-3 |
| NS `nu=1e-3`, T=40 | 1.629e-2 | **1.061e-2** | 3.672e-2 | 1.632e-2 | 3.030e-2 | 1.714e-2 |
| NS `nu=1e-4`, T=20 | 1.798e-1* | **1.597e-1** | 2.493e-1 | 1.694e-1 | 2.161e-1 | 1.602e-1 |
| NS `nu=1e-4`, T=40 | 4.672e-1* | 4.776e-1 | 5.654e-1 | 4.641e-1 | 5.313e-1 | **4.140e-1** |
| Shallow-water, T=40 | 7.970e-3 | **4.725e-3** | 2.876e-2 | 1.486e-2 | 2.101e-2 | 1.230e-2 |

相对旧 PKNO 的变化：

| 任务 | PKNO_v1 RL2 | 相对旧 PKNO |
|---|---:|---:|
| Burgers | 7.694e-3 | 49.0% 变差 |
| NS `nu=1e-3`, T=40 | 1.714e-2 | 5.1% 变差 |
| NS `nu=1e-4`, T=20 | 1.602e-1 | 5.4% 改善 |
| NS `nu=1e-4`, T=40 | 4.140e-1 | 10.8% 改善 |
| Shallow-water, T=40 | 1.230e-2 | 17.2% 改善 |

## 4. 长时间误差行为

下面是 rollout 各步的场 MSE。它用于解释误差如何累积，不替代最终 RL2。

| 任务/模型 | step 1 | step 5 | step 10 | step 20 | step 30 | step 39 |
|---|---:|---:|---:|---:|---:|---:|
| NS `1e-3` PKNO_v1 | 8.49e-5 | 1.10e-4 | 1.16e-4 | 2.73e-4 | 3.84e-4 | 5.71e-4 |
| NS `1e-3` 旧 PKNO | 1.56e-4 | 1.20e-4 | 1.14e-4 | 2.18e-4 | 3.36e-4 | 5.17e-4 |
| NS `1e-4` PKNO_v1 | 1.29e-2 | 5.66e-2 | 1.60e-1 | 5.12e-1 | 8.54e-1 | 1.62e0 |
| NS `1e-4` 旧 PKNO | 3.61e-2 | 9.56e-2 | 2.11e-1 | 5.73e-1 | 1.07e0 | 2.08e0 |
| shallow-water PKNO_v1 | 4.49e-5 | 1.41e-4 | 1.91e-4 | 1.54e-4 | 1.55e-4 | 3.27e-4 |
| shallow-water 旧 PKNO | 2.12e-4 | 2.39e-4 | 2.16e-4 | 2.24e-4 | 2.42e-4 | 3.19e-4 |

解释：

- NS `nu=1e-4` 的改善不是只出现在最后一步，而是从 step 1 到 step 39 都更低，说明软增长约束确实抑制了长期误差放大。
- NS `nu=1e-3` 在前 10 步略有优势，但从约 step 20 开始反超旧 PKNO，说明当前 gate/residual 组合降低了局部误差，却没有完全解决中长时相位或状态漂移。
- shallow-water 前 30 步的 MSE 更低，但末步接近旧 PKNO；RL2 的改善主要来自整体误差和频谱结构，而非末一步单独大幅下降。

## 5. 频谱与梯度诊断

| 任务/模型 | low-band RL2 | mid-band RL2 | high-band RL2 | HF energy-ratio error | gradient RL2 |
|---|---:|---:|---:|---:|---:|
| NS `1e-3` PKNO_v1 | 1.759e-2 | 1.246e2 | 1.605e3 | 1.25e-8 | 5.23e-2 |
| NS `1e-3` 旧 PKNO | 1.636e-2 | 1.321e2 | 1.474e3 | 1.05e-8 | 4.63e-2 |
| NS `1e-4` PKNO_v1 | **4.570e-1** | **1.130** | **1.284** | **4.66e-6** | **8.40e-1** |
| NS `1e-4` 旧 PKNO | 5.179e-1 | 1.282 | 2.393 | 7.55e-5 | 9.15e-1 |
| shallow-water PKNO_v1 | 1.147e-2 | **9.746e-1** | **1.070** | **3.92e-7** | **5.27e-1** |
| shallow-water 旧 PKNO | 9.847e-3 | 1.035 | 14.633 | 9.93e-5 | 1.017 |

这组诊断支持模型设计的局部假设：V1 对 NS `1e-4` 和 shallow-water 的中高频、梯度以及高频能量比例更好，尤其 shallow-water 的 high-band RL2 从 14.63 降到 1.07。相反，Burgers 的 V1 high-band RL2 为 591.1，旧 PKNO 为 90.2；这解释了为什么 Burgers 的整体 RL2 变差。NS `1e-3` 的 mid/high-band 指标很大，通常意味着参考频带能量很小，不能直接按绝对大小解读，但 V1 的梯度和 high-band 仍略差于旧 PKNO。

## 6. 参数量与训练开销

### 参数量

| 模型 | Burgers | NS / shallow-water |
|---|---:|---:|
| KNO | 33,921 | 526,026 |
| iKNO | 116,832 | 608,352 |
| AM-KNO | 286,242 | 571,883 |
| 旧 PKNO | 411,001 | 915,585 |
| PKNO_v1 | 444,025 | 948,353 |

PKNO_v1 相对旧 PKNO 增加约 8.0%（Burgers）和 3.6%（NS/二维场）。它不是参数压缩模型；收益来自条件路径、残差输出和训练课程，而不是更小的网络。

### 500 epoch 墙钟时间

| 任务 | 旧 PKNO 总时长 | PKNO_v1 总时长 | 旧 PKNO 平均/最后 epoch | V1 平均/最后 epoch |
|---|---:|---:|---:|---:|
| Burgers | 0.03 h | 0.04 h | 0.24 / 0.22 s | 0.28 / 0.25 s |
| NS `1e-3`, T=40 | 14.00 h | 10.42 h | 100.83 / 93.18 s | 74.99 / 81.16 s |
| NS `1e-4`, T=20 | 7.29 h | 6.13 h | 52.52 / 53.86 s | 44.11 / 51.23 s |
| NS `1e-4`, T=40 | 12.76 h | 10.64 h | 91.90 / 81.82 s | 76.64 / 86.76 s |
| shallow-water, T=40 | 16.93 h | 18.01 h | 121.89 / 109.27 s | 129.65 / 147.62 s |

这些时间不能简单视为纯模型速度对比：V1 前 100 epoch 使用短 rollout，且 NS 每 epoch 还进行了 validation；旧 PKNO 从第一轮完整 rollout，评估和数据管线也不同。因此 NS 上的总时长下降部分来自 curriculum，shallow-water 的比较还叠加了 `decompose` 不一致。当前上传结果没有记录 V1 的 peak GPU memory 和 inference latency，不能据此声称 V1 更省显存或推理更快。

## 7. 训练过程与稳定性

V1 的训练目标是 MSE 主项（同时有 reconstruction、state、smooth 和特定条件下的 growth 正则），不是 RL2。NS `1e-4` 的 `growth_weight` 为非零，最终 summary 中报告的 growth ceiling 为约 1.09；从 rollout 曲线看没有出现旧 PKNO 那样的快速增长。NS `1e-3` 和 shallow-water 的最终训练均正常完成，没有发现 NaN、Inf、OOM 或 traceback。

验证集上的最佳 RL2 并不总在 final epoch：NS `1e-3` 最低 validation RL2 约为 `1.693e-2`（epoch 488），NS `1e-4` 约为 `4.195e-1`（epoch 492）。本轮最终 summary 使用固定 `checkpoint=final`，且上传目录没有 checkpoint 文件，因此还不能把 validation-best 的 test RL2 作为正式结果。

## 8. 结论

1. PKNO_v1 的核心机制在低黏性 NS 上有效：T=40 RL2 从 `4.641e-1` 降到 `4.140e-1`，改善 10.8%，并且各个 rollout 步和频谱/梯度指标都改善。
2. shallow-water 的整体 RL2 从 `1.486e-2` 降到 `1.230e-2`，频谱高频误差改善尤其明显，但实际 `decompose` 与旧 PKNO 不一致，仍需严格复现实验。
3. Burgers 和 NS `1e-3` 未超过旧 PKNO；前者的高频和梯度误差明显恶化，后者表现为前期较好、后期误差反超。
4. 因 promotion gate 要求四项全部优于旧 PKNO，PKNO_v1 当前应定位为“部分成功的研究分支”，不能替换论文主结果。

## 9. 后续实验建议

按优先级建议：

1. 先用 `decompose=4` 重跑 PKNO_v1 shallow-water，严格对齐旧 PKNO；同时记录 peak memory、单步 inference 和完整 rollout latency。
2. 针对 NS `1e-3` 做最小消融：`gate_max=0`（physics-only）、关闭 residual（direct prediction）、关闭 curriculum，分别判断退化来自 state gate、残差输出还是课程训练。
3. 针对 Burgers 优先检查 residual 输出和频谱损失：当前 high-band/gradient 明显恶化，建议先跑 direct-prediction 消融，再决定是否加入轻量高频/梯度正则；不要改变 `modes=16`。
4. NS `1e-4` 的稳定性方向值得保留，但应至少补充 seed 1/2（或按论文协议的多 seed）确认 `0.414` 不是单 seed 偶然结果。
5. 只有四项在同一协议和多 seed 下都超过旧 PKNO 后，才考虑运行 joint NS 第五实验及论文结果替换。

## 10. 可复核文件

- PKNO_v1 设计：[docs/models/stage3_2_pkno_v1_model_design.md](../docs/models/stage3_2_pkno_v1_model_design.md)
- 实验协议：[docs/experiments/stage3_2_pkno_v1_experiment_design.md](../docs/experiments/stage3_2_pkno_v1_experiment_design.md)
- PKNO_v1 结果：`Latest experimental results uploaded/outputs/stage3_2_pkno_v1/`
- 旧 PKNO 结果：`Latest experimental results uploaded/outputs/stage3_0_param_kno/`
