# IKNO 与 PKNO 基线对比实验报告

日期：2026-08-05

English version: [2026_08_05_ikno_pkno_baseline_comparison.md](2026_08_05_ikno_pkno_baseline_comparison.md)

## 范围与数据来源

本报告汇总以下目录中已完成的实验：

```text
Latest experimental results uploaded/outputs/
```

比较对象包括 KNO、IKNO、AM-KNO、PKNO 与三个 PKNO-U 变体。由于不在本次要求范围内，主表不包含 AM-PKNO。除非另行说明，所有展示的完整运行均使用 seed 42、训练至 epoch 499（即共 500 个 epoch）。误差越低越好。

主指标为完整 rollout 的相对 L2 误差。KNO 的结果取自各运行目录中的 `evaluation_summary.json`；其余模型的结果取自 `metrics.csv` 最后一行。Burgers、NS v=1e-3 与 shallow-water 在各自所在行内使用匹配的样本数和数据划分。NS v=1e-4 是例外：官方 KNO 使用 KoopmanLab 默认的 `ntrain=8000, ntest=200`，而 IKNO、AM-KNO、PKNO 和 PKNO-U 使用 `ntrain=1000, ntest=200`；两者的 200 个测试样本并不相同。因此，NS v=1e-4 中的 KNO 数值只用于结果盘点，不能作为严格的直接比较。T=20 与 T=40 是不同实验，不可跨行比较。

上传配置中的 PKNO-U 变体定义为：

| 变体 | `condition_mode` | 状态嵌入维度 |
|---|---|---:|
| PKNO-U A | `physical_only` | 16 |
| PKNO-U B | `physical_compact_state` | 16 |
| PKNO-U C | `physical_gated_state` | 16 |

它们并非仅改变一个变量的 Stage 3 PKNO 消融：除条件输入路径之外，其状态嵌入也从 Stage 3 PKNO 默认的 64 改为了 16。因此，应将其视为替代模型变体，而非只隔离 gate 作用的严格消融。

## 主要结果

完整 rollout 相对 L2 误差。粗体表示该行已完成实验中的最低值。空白表示没有完成的 500-epoch 结果，按要求不以部分训练结果填补。

| 数据集与预测时域 | KNO | IKNO | AM-KNO | PKNO | PKNO-U A | PKNO-U B | PKNO-U C |
|---|---:|---:|---:|---:|---:|---:|---:|
| Burgers, T=1 | 7.887e-3 | 9.602e-3 | 1.600e-2 | **5.163e-3** | 1.475e-2 | 1.561e-2 | 1.471e-2 |
| NS v=1e-3, T=40 | 1.629e-2 | **1.061e-2** | 3.672e-2 | 1.632e-2 | 5.314e-2 | 4.407e-2 | 4.961e-2 |
| NS v=1e-4, T=20 | 1.798e-1* | **1.597e-1** | 2.493e-1 | 1.694e-1 | 2.749e-1 | 2.679e-1 | 2.759e-1 |
| NS v=1e-4, T=40 | 4.672e-1* | 4.776e-1 | 5.654e-1 | **4.641e-1** | 5.639e-1 | 5.850e-1 | |
| Shallow-water, T=40 | 7.970e-3 | **4.725e-3** | 2.876e-2 | 1.486e-2 | 1.136e-2 | 1.284e-2 | |

`*` 表示官方 KNO 的 NS v=1e-4 数据划分不匹配；不可据此计算 KNO 相对于其他模型的改善。PKNO-U C 在 NS v=1e-4 T=40 的 epoch 420 和 shallow-water 的 epoch 343 存有结果，但均非完整运行，故保留空白。此前 Stage 3 PKNO 的 shallow-water 运行在 epoch 11 结束，同样被排除；此处采用稳定完成的 `lr=5e-5, decompose=4, delta_scale=0.005` 运行。

## 配置与计算成本

### 指标定义与限制

- 表中每个完整运行均使用 500 个 epoch 和 seed 42。非 KNO 模型的 `seconds` 是 `metrics.csv` 中单个 epoch 的墙钟时间，包含该 epoch 的训练与测试评估。表格同时报告 500 个已记录 epoch 的时间总和以及最后一个 epoch 的时间；两者都不是纯训练 step 的基准。
- 官方 KNO 的 stage-0 `metrics.csv` 未记录可与之对应的 epoch 墙钟时间。其 `complexity.csv` 记录了参数量、峰值显存、单步推理延迟和完整 rollout 延迟。其他实现没有写入峰值显存或推理延迟，故这些格子有意留空，不进行估计。
- 所有时间均受运行环境影响。shallow-water 的 IKNO 在 batch size 5 的 smoke test 耗尽 47.54 GB GPU 显存后，使用 batch size 3 完成；AM-KNO、PKNO 和 PKNO-U 的完整 shallow-water 运行均使用 batch size 5。因此，涉及 shallow-water IKNO 的 epoch 耗时并未按 batch size 归一化。
- NS v=1e-4 中 KNO 使用 8000/200 划分，其他方法使用 1000/200 划分。这不仅影响精度，也影响 epoch 成本，因此该两行的 KNO 耗时不具备相同协议下的可比性。

### 模型设置

所有模型均使用 observable/operator size `O=32`；对采用截断模态算子的模型，使用 Fourier modes `modes=16`；优化器为 Adam，`weight_decay=1e-4`、`gamma=0.5`、`step_size=100`，并采用下表所示的、与数据集对应的自回归时域。AM-KNO 使用全部 FFT 模态（`max_modes=0`），因此其配置中的 `modes=16` 不是该模型的算子频率截断。

| 模型 | 算子/字典设计 | 关键结构超参数 | 学习率：Burgers / NS 1e-3 / NS 1e-4 / shallow |
|---|---|---|---|
| KNO | 官方 KoopmanLab：每个保留 Fourier 模态对应一个固定复矩阵 | `O=32`，modes 16，`decompose=8` | `1e-3 / 1e-3 / 1e-3 / 1e-3` |
| IKNO | 固定 Fourier Koopman 算子，搭配逐点可逆残差耦合字典 | `O=32`，modes 16，`decompose=4`，`koopman_power=2`，4 个 INN block，INN 隐层宽度 128 | `1e-3 / 1e-3 / 1e-3 / 1e-3` |
| AM-KNO | 频率条件化的算子生成器；二维算子采用 rank 1 因式分解 | `O=32`，全部 FFT 模态，`decompose=8`，生成器深度 2、隐藏宽度 128 | `1e-3 / 5e-4 / 3e-4 / 2e-4` |
| PKNO | 由物理元数据与 64 维状态摘要条件化的共享 Koopman 矩阵字典 | `O=32`，modes 16，字典深度 2 / 宽度 128，状态嵌入 64；`decompose=8`，shallow 为 4；delta scale `.05`，shallow 为 `.005` | `1e-3 / 5e-4 / 5e-4 / 5e-5` |
| PKNO-U A | PKNO-U，仅使用物理元数据 | `O=32`，modes 16，字典深度 2 / 宽度 128，状态嵌入 16，`decompose=8` | `1e-3 / 5e-4 / 3e-4 / 5e-5` |
| PKNO-U B | PKNO-U，物理元数据加紧凑状态路径 | 与 A 相同；`condition_mode=physical_compact_state` | `1e-3 / 5e-4 / 3e-4 / 5e-5` |
| PKNO-U C | PKNO-U，物理元数据加门控状态路径 | 与 A 相同；`condition_mode=physical_gated_state` | `1e-3 / 5e-4 / 3e-4 / 5e-5` |

共享 trainer 中，Burgers 和两种 Navier-Stokes 黏度设置均使用 `ntrain/ntest=1000/200`。Shallow-water 使用 `900/100`，空间场大小为 128 x 128，`T_in=10`、`T_out=40`。KNO 在 NS v=1e-4 使用其官方默认的 `8000/200` 协议，如前所述。Burgers 是单步 rollout（`T=1`）；所列 Navier-Stokes 和 shallow-water 配置均为 `T_in=10`，`T_out` 取对应行所示时域。

### 参数量与实际训练时间

参数量为相应完整运行写入的精确值。`500 epochs 总时间（h）/ 最后 epoch（s）` 分别为 500 条 `seconds` 记录之和换算的小时数，以及最后一行的 `seconds`；其中包含训练和评估时间。空白表示该运行没有该测量记录，不表示没有计算成本。PKNO-U C 的 T=40 和 shallow-water 运行未完成，对应格子留空。

| 数据集与预测时域 | 测量 | KNO | IKNO | AM-KNO | PKNO | PKNO-U A | PKNO-U B | PKNO-U C |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Burgers, T=1 | 参数量 | 33,921 | 116,832 | 286,242 | 411,001 | 469,305 | 474,453 | 474,470 |
|  | 500 epochs 总时间（h）/ 最后 epoch（s） | | 0.042 / 0.302 | 0.025 / 0.179 | 0.033 / 0.222 | 0.090 / 0.659 | 0.087 / 0.603 | 0.085 / 0.642 |
| NS v=1e-3, T=40 | 参数量 | 526,026 | 608,352 | 571,883 | 915,585 | 1,135,553 | 1,147,721 | 1,147,738 |
|  | 500 epochs 总时间（h）/ 最后 epoch（s） | | 9.655 / 68.224 | 9.580 / 67.818 | 14.004 / 93.185 | 27.465 / 192.557 | 29.402 / 223.273 | 29.876 / 200.696 |
| NS v=1e-4, T=20 | 参数量 | 526,026 | 608,352 | 571,883 | 915,585 | 1,135,553 | 1,147,721 | 1,147,738 |
|  | 500 epochs 总时间（h）/ 最后 epoch（s） | | 4.996 / 36.369 | 4.334 / 54.657 | 7.294 / 53.860 | 14.046 / 98.051 | 14.498 / 106.184 | 14.659 / 110.645 |
| NS v=1e-4, T=40 | 参数量 | 526,026 | 608,352 | 571,883 | 915,585 | 1,135,553 | 1,147,721 | |
|  | 500 epochs 总时间（h）/ 最后 epoch（s） | | 9.687 / 68.990 | 9.189 / 60.261 | 12.765 / 81.822 | 29.025 / 209.556 | 29.013 / 230.714 | |
| Shallow-water, T=40 | 参数量 | 526,026 | 608,352 | 571,883 | 915,585 | 1,135,553 | 1,147,721 | |
|  | 500 epochs 总时间（h）/ 最后 epoch（s） | | 31.184 / 220.730（batch 3） | 26.168 / 186.569（batch 5） | 16.929 / 109.266（batch 5） | 53.266 / 388.030（batch 5） | 55.604 / 398.075（batch 5） | |

### 已记录的 KNO 推理与显存指标

这是上传结果中唯一直接记录的部署阶段测量。`Rollout (ms)` 是指定预测时域内一次完整自回归评估的时间，`Step (ms)` 为相应单步测量。参数量对应上表 KNO 列。

| KNO 数据集与预测时域 | 峰值 GPU 显存（GB） | Step（ms） | Rollout（ms） |
|---|---:|---:|---:|
| Burgers, T=1 | 0.03224 | 2.108 | 2.108 |
| NS v=1e-3, T=40 | 0.15244 | 3.090 | 123.602 |
| NS v=1e-4, T=20 | 0.10999 | 2.752 | 55.037 |
| NS v=1e-4, T=40 | 0.15244 | 3.076 | 123.050 |
| Shallow-water, T=40 | 0.27408 | 3.667 | 146.677 |

### 成本解读

IKNO 相对紧凑：相对于 KNO，它在一维和二维中均约多出 82k 参数，同时在 NS v=1e-3 与 shallow-water 获得了已完成实验中的最佳精度。当前 PKNO 在二维中的参数量比 KNO 高 74%（915,585 对 526,026），PKNO-U 变体则高约 116-118%。相应地，它们的最后 epoch 时间通常更长，尤其在 T=40 的 Navier-Stokes 实验中。就当前单条件设置而言，这部分成本尚未换来相对于 KNO 的一致精度提升。

在 shallow-water 上，PKNO 的实际最后 epoch 时间虽有更多参数却低于 AM-KNO。这应被视为当前实现路径下的观测，而不是一般性复杂度结论：AM-KNO 的全频率生成器和因式分解算子具有不同计算模式，且这里的数值还包含数据加载和评估。公平的效率结论需要在完全相同的 batch size、精度、设备和测试子集下，统一记录预热后的训练吞吐、峰值已分配显存与批量推理延迟。

## 结果支持的结论

### IKNO

IKNO 在五行中的三行取得最强的已完成非 KNO 结果：NS v=1e-3 T=40、NS v=1e-4 T=20 与 shallow-water T=40。其 shallow-water 相对 L2 比 KNO 低 40.7%。但它并非在所有任务上占优：在 Burgers 上差于 KNO。由于数据划分不一致，NS v=1e-4 与 KNO 的比较不成立。

### PKNO 相对于 AM-KNO

PKNO 在每一行匹配结果中均优于 AM-KNO：Burgers 67.7%，NS v=1e-3 T=40 为 55.6%，NS v=1e-4 T=20 为 32.1%，NS v=1e-4 T=40 为 17.9%，shallow-water 为 48.3%。这是内部一致的结果：共享字典加条件化 Fourier Koopman 构造显著优于当前仅频率条件化的 AM-KNO 基线。

### PKNO 相对于 KNO

更强的结论，即 PKNO 在长 rollout 上普遍优于官方 KNO，目前尚未建立。

- Burgers：PKNO 比 KNO 好 34.5%，但这是单步任务，并不能检验长时域稳定性。
- NS v=1e-3 T=40：PKNO 与 KNO 的完整 rollout 相对 L2 基本持平（1.632e-2 对 1.629e-2）。
- NS v=1e-4 T=20 与 T=40：当前 KNO 的 8000 训练样本和不同的 200 测试样本，与其他模型使用的 1000/200 协议不一致；这两行不能用于 KNO 与 PKNO 的结论。
- Shallow-water T=40：PKNO 比 KNO 差 86.5%，尽管明显优于 AM-KNO。

因此，当前结果支持较窄的表述：PKNO 修复了 AM-KNO 引入的大部分性能损失，并在 NS v=1e-3 上与 KNO 具有竞争力；但尚未证明其在困难 PDE 上具有可靠且显著的长 rollout 优势。

## 长 rollout 诊断

下表取自 `rollout_error_by_step.csv` 的第一行和最后一行。这些值是逐步 MSE，只应在同一数据集和时域内比较。增长倍数为最后一步 MSE 除以第一步 MSE。

| 数据集 | 模型 | 第 1 步 MSE | 最后一步 MSE | 增长倍数 |
|---|---|---:|---:|---:|
| NS v=1e-3, T=40 | KNO | 7.282e-5 | 6.059e-4 | 8.32x |
|  | IKNO | 1.344e-4 | **2.238e-4** | **1.67x** |
|  | PKNO | 1.713e-4 | 5.170e-4 | 3.02x |
| NS v=1e-4, T=40 | KNO | 3.575e-2 | **1.876e+0** | **52.48x** |
|  | IKNO | 3.303e-2 | 2.491e+0 | 75.42x |
|  | PKNO | **2.125e-2** | 2.076e+0 | 97.68x |
| Shallow-water, T=40 | KNO | **3.625e-5** | 1.167e-4 | 3.22x |
|  | IKNO | 3.535e-5 | **4.505e-5** | **1.27x** |
|  | PKNO | 2.063e-4 | 3.188e-4 | 1.55x |

PKNO 在长 rollout 上有一个积极信号：NS v=1e-3 中，其第一步 MSE 差于 KNO，但最后一步 MSE 低 14.7%，增长倍数也更低。由于 PKNO 在 rollout 前段损失了过多精度，完整 rollout 相对 L2 仍与 KNO 持平。

其他两个 T=40 情形无法建立所期望的结论。NS v=1e-4 中，PKNO 从自身第一步到最后一步的误差增长 97.68 倍，但这里的 KNO 曲线来自不同划分，不能作为比较对象。Shallow-water 中，PKNO 的增长倍数较低，但其初始误差约为 KNO 的 5.7 倍；稳定的传播无法补偿不够准确的初始预测。

## PKNO 未出现显著长时域优势的原因

1. **当前每个训练 run 内物理条件是常量。** Stage 3 loader 对同一个数据文件中的所有样本重复同一个黏度、网格和时间条件向量。因此，PKNO 并未在跨样本变化的物理参数族上训练。其参数化 Koopman 映射几乎没有机会学习有用的物理条件响应，只能利用由历史状态计算出的状态摘要。

2. **目标函数没有显式强调后期误差或稳定性。** 共享 trainer 只对 rollout 各步的预测 MSE 求和。它没有提高后期 step 权重、最后时刻损失、rollout 增长惩罚、谱半径约束、守恒约束或潜在 Koopman 一致性损失。模型可以取得有竞争力的平均损失，同时在后期持续累积误差。

3. **比较对象是强且专用的 KNO。** KNO 对每个训练数据集独立拟合每个频率的固定复矩阵。PKNO 用共享字典和条件化算子生成器替代了这种直接自由度。当物理条件不在样本间变化时，额外结构可能成为优化负担，而非新增信息。当前 KNO 的 NS v=1e-4 还使用不同的 8000-sample 协议，必须用 1000/200 划分重跑后才能成为有效 PKNO 基线。

4. **当前条件化变体没有解决该问题。** 已完成行中，PKNO-U A/B/C 均持续差于 Stage 3 PKNO。B 通常在 Navier-Stokes 上优于 A，说明状态信息可能有帮助，但两者均未达到 Stage 3 PKNO 或 IKNO 的已完成匹配结果。这不支持将当前条件化/门控路径作为已验证提升来源。

5. **证据仅有单一随机种子，且实现混合。** 所有结果均为 seed 42。KNO 通过官方 wrapper 评估，其余模型通过共享 PyTorch rollout trainer 评估。完整相对 L2 标量结果有参考价值，但没有重复随机种子和统一 evaluator 校验，微小差距不能作为可发表结论。

## 当前 PKNO 结果是否令人满意？

作为中间消融结果可以接受，但不足以作为预期中的主结果。

模型已经满足较弱目标：实质性优于当前 AM-KNO 实现，并在 NS v=1e-3 上呈现有希望的误差增长特征。但它尚未达到更强目标，即在多个困难 PDE 上相对于 KNO 获得清晰、可重复的长 rollout 优势。Shallow-water 是最明确的反例；NS v=1e-4 必须先完成匹配的 KNO 重跑，才能作为支持或反对证据。

## 建议的下一步实验

1. 在混合黏度上训练真正的联合 Navier-Stokes 模型，至少将 v=1e-3 与 v=1e-4 放入同一训练集，保留黏度条件向量，并以黏度插值/外推测试。只有这一实验能够检验参数化 Koopman 家族。

2. 加入 rollout 感知目标：随时间步增加的权重、最后一步相对 L2 项，以及对生成 Koopman 矩阵的稳定性正则。报告平均相对 L2 和最后一步相对 L2。

3. 进行严格的 PKNO 消融，其中只修改条件输入路径；observable 维度、字典深度/隐藏宽度、状态嵌入维度、优化器、时域和随机种子集合均保持一致。

4. 首先以显式 `--ntrain 1000 --ntest 200` 在 NS v=1e-4 上重跑官方 KNO；随后在一个 evaluator 下，以至少三个随机种子重跑 KNO、IKNO、PKNO 与选定的 PKNO-U 变体。报告均值和标准差，并保留逐步误差曲线。

5. 在强调高频结论前，应在完整测试集上补充谱指标，而不是使用当前仅针对首个评估 batch 的统计。
