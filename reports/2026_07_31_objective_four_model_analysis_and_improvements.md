# 2026-07-31 KNO、AM-KNO、PKNO、AM-PKNO 客观比较与改进报告

## 范围与总体判断

本报告覆盖 First_set_of_valid_data_statistics_7_31_13_33 中 KNO、AM-KNO、
PKNO 以及 AM-PKNO 的所有可见记录，包括未完成的 AM-PKNO。报告严格区分
“运行完成”“数值有效”“科学上可公平比较”。全部完成运行仅使用 seed 42 和
最后记录 epoch，因此每一项排序都只是点估计，不是统计检验结论。

核心发现：

1. KNO 是当前最强的固定基线：浅水最佳，NS nu=1e-3 点估计最佳，NS nu=1e-4
   与 PKNO 几乎持平。
2. AM-KNO 是有价值的负对照。其全频、仅频率生成器在每个完成任务上都劣于
   KNO，Full Rel. L2 退化 21.0% 到 260.9%。
3. PKNO 在每个完成任务上都显著改善 AM-KNO：Burgers 67.7%、NS nu=1e-3
   55.6%、NS nu=1e-4 T=40 17.9%、浅水 48.3%。这支持完整 PKNO 设计相对
   AM-KNO 的收益，但不支持其对 KNO 的普适优势。
4. 完成的 AM-PKNO 介于 AM-KNO 与 PKNO 之间：相对 AM-KNO 有改善，但在
   Burgers 和两个完成的 T=40 NS 上都不如 PKNO 和 KNO。浅水尚未完成，不能
   作为最终结论。
5. 最紧迫的实验债务是统一协议下的多 seed 复跑。NS 上 KNO-PKNO 的差异不足
   1%，单 seed 无法确认其方向是否稳定。

## 完成度与记录状态

| 方法 | Burgers | NS nu=1e-3，T=40 | NS nu=1e-4，T=20 | NS nu=1e-4，T=40 | Shallow water，T=40 |
|---|---|---|---|---|---|
| KNO | 完成 | 完成 | 完成 | 完成 | 完成 |
| AM-KNO | 完成 | 完成 | 完成 | 完成 | 完成 |
| PKNO | 完成 | 完成 | 完成 | 完成 | 稳定化重跑完成；原 r=8 失败 |
| AM-PKNO | 完成 | 完成 | metrics.csv 为空 | 完成 | 部分完成：epoch 332/500，缺 rollout/spectral 文件 |

AM-PKNO T=20 是缺失记录，不能解释为负的数值结果。PKNO 的原浅水 r=8
运行是真实的不稳定记录：epoch 7 最佳 test full 约为 3.112e-2，之后误差
抬升，epoch 12 触发 non-finite loss。用于正式对比的是 r=4、lr=5e-5、
delta_scale=0.005、max_grad_norm=0.1 的稳定化重跑。

## Full Rel. L2 总览

| 数据集 | KNO | AM-KNO | PKNO | AM-PKNO | 完成运行中的最低点估计 |
|---|---:|---:|---:|---:|---|
| Burgers | 7.887e-3 | 1.600e-2 | **5.163e-3** | 1.265e-2 | PKNO |
| NS nu=1e-3，T=40 | **1.629e-2** | 3.672e-2 | 1.632e-2 | 3.030e-2 | KNO；PKNO +0.15% |
| NS nu=1e-4，T=20 | 1.798e-1 | 2.493e-1 | **1.694e-1** | 缺失 | PKNO |
| NS nu=1e-4，T=40 | 4.672e-1 | 5.654e-1 | **4.641e-1** | 5.313e-1 | PKNO |
| Shallow water，T=40 | **7.970e-3** | 2.876e-2 | 1.486e-2 | 2.149e-2* | KNO |

T=20 是 horizon 诊断，不能与 T=40 混合。带 * 的 AM-PKNO 浅水数值来自未完成
epoch 332，仅可观察训练进度。

## 逐步比较

### KNO 到 AM-KNO

将 KNO 的 retained-mode 矩阵表替换为全频、仅频率的生成器后，所有完成任务
均变差：

| 数据集 | AM-KNO 相对 KNO 的 Full Rel. L2 变化 | 解释 |
|---|---:|---|
| Burgers | +102.8% | 生成器丢失显著的一步精度 |
| NS nu=1e-3，T=40 | +125.4% | 长 rollout 与频谱质量均显著退化 |
| NS nu=1e-4，T=20 | +38.6% | 短 horizon 已出现退化 |
| NS nu=1e-4，T=40 | +21.0% | 困难流动下仍劣于 KNO |
| Shallow water | +260.9% | 最大退化，尽管训练完整结束 |

所以 AM-KNO 应作为信息量很高的消融/负对照，而不能描述为更强的独立 baseline。
另外，它与 KNO/PKNO 同时存在“全频与 modes=16”和“矩阵参数化方式”的差别，
本比较不能归因给单一因素。

### AM-KNO 到 PKNO

| 数据集 | AM-KNO | PKNO | PKNO 降幅 | 可支持的结论 |
|---|---:|---:|---:|---|
| Burgers | 1.600e-2 | 5.163e-3 | 67.7% | 强的点估计收益 |
| NS nu=1e-3，T=40 | 3.672e-2 | 1.632e-2 | 55.6% | 强的长 rollout 收益 |
| NS nu=1e-4，T=20 | 2.493e-1 | 1.694e-1 | 32.1% | horizon 诊断收益 |
| NS nu=1e-4，T=40 | 5.654e-1 | 4.641e-1 | 17.9% | 仍有收益，但绝对误差高 |
| Shallow water | 2.876e-2 | 1.486e-2 | 48.3% | 有收益，但 r=8 对稳定化 r=4 |

PKNO 这一步同时引入/恢复了按 mode 的基矩阵、条件化传播和混合字典。因此，它
验证的是完整 PKNO 相对 AM-KNO，而不能将收益分别归因给任一组件。

### PKNO 到 AM-PKNO

AM-PKNO 结合了 AM 风格的全频条件化分解生成器与 PKNO 风格的字典/条件信息。
它恢复了一部分 AM-KNO 性能，但没有达到 PKNO：

| 数据集 | AM-KNO | PKNO | AM-PKNO | AM-PKNO 相对 AM-KNO | AM-PKNO 相对 PKNO |
|---|---:|---:|---:|---:|---:|
| Burgers | 1.600e-2 | **5.163e-3** | 1.265e-2 | -20.9% | +145.0% |
| NS nu=1e-3，T=40 | 3.672e-2 | **1.632e-2** | 3.030e-2 | -17.5% | +85.7% |
| NS nu=1e-4，T=40 | 5.654e-1 | **4.641e-1** | 5.313e-1 | -6.0% | +14.5% |
| Shallow water，T=40 | 2.876e-2 | **1.486e-2** | 2.149e-2* | -25.3%* | +44.6%* |

AM-PKNO 的浅水行仅表示训练过程进展。当前证据不支持在论文中用 AM-PKNO 替换
PKNO。

## 诊断指标

仓库已经说明 spectral_metrics.csv 仅针对首个 evaluation batch。它适合定位
问题和作为辅助诊断，不能成为论文主指标。

- Burgers：PKNO 的 gradient/high-band 误差最低，分别为 2.681e-2/9.016e1；
  KNO 为 5.864e-2/1.771e2，AM-KNO 为 6.679e-2/1.882e2。这与主排序一致。
- NS nu=1e-3：PKNO 相对 AM-KNO 将 low/mid/high 从
  3.830e-2 / 6.564e2 / 6.625e3 降至
  1.636e-2 / 1.321e2 / 1.474e3，gradient 从 1.286e-1 降至 4.634e-2。
  KNO 的低频误差相近、高频误差略低于 PKNO；主指标中 KNO 也以 0.15% 领先。
- NS nu=1e-4：PKNO 改善 AM-KNO 的低频与 gradient，但中/高频略差，不能宣称
  普适的高频改进。
- Shallow water：PKNO gradient 为 1.017，略好于 AM-KNO 的 1.064，却显著
  高于 KNO 的 0.434，与 KNO 在主指标上的优势相一致。部分完成的 AM-PKNO
  没有频谱诊断文件。

## 成本与公平性

| 方法 | Burgers 参数量 | 2D 参数量 | 主要差别 |
|---|---:|---:|---|
| KNO | 33,921 | 526,026 | 官方固定 retained-mode 矩阵 |
| AM-KNO | 286,242 | 571,883 | 全频、分解的仅频率生成器 |
| PKNO | 411,001 | 915,585 | retained-mode 基矩阵、条件修正、混合字典 |
| AM-PKNO | 382,329 | 696,705 | 全频、条件化分解生成器与字典 |

PKNO 的参数量是 AM-KNO 的 1.44 倍（Burgers）和 1.60 倍（2D）。所以
AM-KNO 到 PKNO 表明的是有价值的精度-容量设计，而非同参数量优势。AM-PKNO
参数少于 PKNO，但完成的 NS 每 epoch 更慢：nu=1e-3 为 202.6 s 对 93.2 s，
nu=1e-4 为 202.7 s 对 81.8 s。这只能作为同类 PyTorch 实现下的迹象。KNO
保存的是不同的计时产物，不能直接与其排名。

还存在以下可比性限制：

- KNO NS 保存的 args 中 ntrain/ntest 为 null，而 AM-KNO/PKNO 记录为
  1000/200；应明确官方 wrapper 实际使用的数据切分。
- 所有正式主结果都是单 seed。
- 浅水 PKNO 使用 r=4，KNO/AM-KNO 使用 r=8。
- 相对 L2 名称已对齐，但提交前仍应固化一个共享 evaluator 命令和完整测试集
  聚合过程。

## 改进建议，按优先级排序

### 1. 建立统一的三 seed 主结果

对 Burgers、NS nu=1e-3 T=40、NS nu=1e-4 T=40 的 KNO、AM-KNO、PKNO
分别运行三个 seed，明确使用相同切分、归一化和最终 evaluator。报告 Full
Rel. L2 的均值加减标准差。这一步价值最高，因为它直接决定 NS 上不足 1% 的
KNO-PKNO 差异是否真实。

### 2. 补全 KNO 的复现元数据

让 KNO wrapper 在 args.json 或 evaluation_summary.json 中保存 ntrain、ntest、
数据切分索引或 hash、归一化信息、evaluator 版本及完整测试集聚合选项。新的
matched relative L2 是关键进展，但元数据对称性仍不够。

### 3. 隔离 PKNO 的组件收益

在 Burgers 和 NS nu=1e-3 训练以下尽量参数匹配的变体：

- 仅 mode-indexed base；
- base 加共享字典；
- base 加条件化修正；
- 完整 PKNO。

尽可能固定 O=32、modes=16、r=8、优化器、数据切分与参数预算。只有这样才能
区分收益来自字典、条件传播还是额外容量。

### 4. 将浅水视为算子稳定性问题

既有稳定性记录表明，单纯减小 lr、修正规模和 clipping 不能让 r=8 稳定。
应逐 rollout step 记录 latent norm、DeltaK Frobenius/spectral norm 与 decoder
输出范围。建议依次尝试：DeltaK 的谱范数/Frobenius 范数约束；有界残差形式
K=K0+alpha*DeltaK；每次 Koopman 更新后的谱阻尼；以及 r in {2,4,6,8} 的
同预算扫描。模型结论只能在相同 r 下比较。

### 5. 改进低黏度的 horizon 曲线

对 NS nu=1e-4 加入后期步数加权的 rollout loss、逐步误差斜率监控，以及
scheduled sampling 或多步训练窗口。保留 T=20/T=40 配对，以判断修改是否
改善 horizon scaling，而不是只优化短预测。

### 6. 先低成本诊断 AM-PKNO

补齐其 SWE 和 T=20 记录。之后在 NS nu=1e-3 上固定其他配置，扫描
factorized rank {1,2,4} 与 max_modes {16,24,0}。这能区分问题来自全频设计、
rank=1 分解，还是优化/正则化；在此之前不应启动大规模 AM-PKNO 网格。

### 7. 将频谱指标聚合到完整测试集

对全部测试样本聚合 low/mid/high 与 gradient，并明确时间维的聚合规则；将
0--33%、33--66%、66--100% 频带边界写入元数据。这样才能把现有诊断升级为
可提交的辅助证据。

## 来源

主要数值为下列目录 metrics.csv 的最后一行：
- First_set_of_valid_data_statistics_7_31_13_33/outputs/stage0_kno_baseline/
- First_set_of_valid_data_statistics_7_31_13_33/outputs/stage1_0_am_kno/
- First_set_of_valid_data_statistics_7_31_13_33/outputs/stage3_0_param_kno/
- First_set_of_valid_data_statistics_7_31_13_33/outputs/stage4_0_am_pkno/

解释和协议依据：docs/metrics/high_frequency_metrics.md、
docs/experiments/stage1_0_am_kno_design.md、
docs/experiments/stage3_0_pkno_design.md、
docs/experiments/stage3_0_shallow_water_stability_notes.md、以及
docs/experiments/stage4_0_am_pkno_design.md。
