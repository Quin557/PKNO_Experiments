# 2026-07-31 PKNO 论文导向实验报告

## 结论摘要

本报告整理目录 First_set_of_valid_data_statistics_7_31_13_33 中最新的
KNO、AM-KNO 与 PKNO 结果。论文中的 Stage 3 ParamKNO 以下均记作 PKNO；
AM-PKNO 因实验矩阵尚不完整，不纳入当前论文主张。

相较于 2026-07-29 临时报告，最重要的更新是 KNO 的重跑现在已经直接导出
Step Rel. L2 和 Full Rel. L2。因此，旧报告和论文草稿中 NS/SWE 的 KNO
估算值必须全部替换。

当前可被数据支持的核心表述为：

- PKNO 在 Burgers 上三者最佳：比 KNO 的 Full Rel. L2 低 34.5%，比
  AM-KNO 低 67.7%。
- PKNO 在 NS nu=1e-3 上相对 AM-KNO 降低 55.6%，但比直接测得的 KNO
  高 0.15%。这应写成“与 KNO 基本持平”，不能写成“超过 KNO”。
- PKNO 在 NS nu=1e-4 上比 AM-KNO 低 17.9%，比 KNO 低 0.66%；该差异
  小于 1%，单个 seed 不能支持稳定排序结论。
- PKNO 在浅水上比 AM-KNO 低 48.3%，但比 KNO 高 86.5%。这是有效的负结果，
  表明当前 PKNO 并非在所有 PDE 上优于固定 KNO。

因此，最稳妥的论文结论是：在当前同口径、单 seed 的结果中，PKNO 在四项
已完成任务上都优于 AM-KNO，并且与固定 KNO 具有竞争力，但尚未证明对 KNO
具有普适优势。

## 数据来源、口径与可比性

全部数值取每个运行 metrics.csv 的最后一行，不使用测试集上的最优 epoch：
KNO 为 epoch 500，PyTorch 方法为 epoch 499。所有正式运行均为 seed 42，
所以表中是点估计，不是具有统计显著性的均值。

| 任务 | 预测协议 | 关键配置 | 状态 |
|---|---|---|---|
| Burgers | 一步场预测 | O=32；KNO/PKNO modes=16、r=8；AM-KNO 全频、rank=1 | 完成 |
| NS nu=1e-3 | t_in=10，t_out=40，64x64 | O=32、r=8；AM-KNO 全频、rank=1；PyTorch 训练/测试为 1000/200 | 完成 |
| NS nu=1e-4 | t_in=10，t_out=40，64x64 | 同上 | 完成 |
| Shallow water | t_in=10，t_out=40，128x128 | KNO/AM-KNO r=8；稳定化 PKNO r=4；训练/测试为 900/100 | 完成，但有 r 不一致限制 |

KNO 使用官方 KoopmanLab wrapper。AM-KNO 是仅按频率生成的
K_k=G(e(k))。PKNO 使用按 mode 的基矩阵、依赖条件/状态的修正以及共享混合
字典。因此这是完整设计之间的比较，不是参数量完全匹配的消融。

浅水 PKNO 使用稳定化运行
pkno_shallow_water_o32_m16_r8_t40_ep500_seed42_lr1e4_ds1e2；实际参数为
r=4、lr=5e-5、delta_scale=0.005、max_grad_norm=0.1。名义的 r=8、
lr=2e-4、delta_scale=0.02 运行在 epoch 11 后出现非有限 loss。因此，浅水
PKNO 数值在正文、表注和结论中都必须保留此限制。

## 主结果

Step Rel. L2 是 rollout 各步的平均归一化误差；Full Rel. L2 是用于排序的
完整序列归一化误差。数值越低越好。

| 数据集 | 方法 | Step Rel. L2 | Full Rel. L2 | 相对 KNO | 相对 AM-KNO |
|---|---|---:|---:|---:|---:|
| Burgers | KNO | 7.887e-3 | 7.887e-3 | -- | -50.7% |
|  | AM-KNO | 1.600e-2 | 1.600e-2 | +102.8% | -- |
|  | **PKNO** | **5.163e-3** | **5.163e-3** | **-34.5%** | **-67.7%** |
| NS nu=1e-3，T=40 | **KNO** | **1.551e-2** | **1.629e-2** | -- | -55.6% |
|  | AM-KNO | 3.503e-2 | 3.672e-2 | +125.4% | -- |
|  | PKNO | 1.570e-2 | 1.632e-2 | +0.15% | **-55.6%** |
| NS nu=1e-4，T=40 | KNO | 4.118e-1 | 4.672e-1 | -- | -17.4% |
|  | AM-KNO | 5.168e-1 | 5.654e-1 | +21.0% | -- |
|  | **PKNO** | **4.022e-1** | **4.641e-1** | **-0.66%** | **-17.9%** |
| Shallow water，T=40 | **KNO** | **7.909e-3** | **7.970e-3** | -- | -72.3% |
|  | AM-KNO | 2.856e-2 | 2.876e-2 | +260.9% | -- |
|  | PKNO | 1.483e-2 | 1.486e-2 | +86.5% | **-48.3%** |

表中小数精度仅服务于复现。没有置信区间；NS nu=1e-3 的 0.15% 和
NS nu=1e-4 的 0.66% KNO-PKNO 差异都不能在单 seed 下解释为稳健优势。

## 对 PKNO 有利但不夸大的任务分析

### Burgers

这是当前最干净的 PKNO 结果。PKNO Full Rel. L2 为 5.163e-3，较 KNO
降低 34.5%，较 AM-KNO 降低 67.7%。其预测 MSE 为 3.115e-5，接近 KNO 的
2.311e-5，远低于 AM-KNO 的 1.267e-4。首个 evaluation batch 的低频、
高频和梯度诊断误差也是三者最低：6.345e-3、9.016e1 和 2.681e-2。频谱指标
只能作为诊断，但方向与主指标一致。

### NS nu=1e-3

PKNO 将 AM-KNO 的 Full Rel. L2 从 3.672e-2 降至 1.632e-2，降幅 55.6%；
mean-step MSE 与 KNO 接近。直接测得的 KNO Full Rel. L2 为 1.629e-2，仍比
PKNO 低 0.15%。正文应写“PKNO 与 KNO 在单 seed 下基本持平”，而不是
“PKNO 超过 KNO”。

频谱诊断同样支持 PKNO 相对 AM-KNO 的机制性收益：低/中/高频误差从
3.830e-2 / 6.564e2 / 6.625e3 降为 1.636e-2 / 1.321e2 / 1.474e3；梯度
误差从 1.286e-1 降为 4.634e-2。

### NS nu=1e-4

PKNO 将 AM-KNO Full Rel. L2 从 5.654e-1 降至 4.641e-1，降幅 17.9%，并
略低于 KNO 的 4.672e-1。绝对误差仍然很高。适当表述是“PKNO 缓解了相对
AM-KNO 的低黏度退化”，不能称已解决长 rollout 稳定性。

T=20 仅为 horizon 诊断，不能混入 T=40 主表。KNO / AM-KNO / PKNO 的
Full Rel. L2 分别为 1.798e-1 / 2.493e-1 / 1.694e-1；PKNO 相对 AM-KNO
降低 32.1%，相对 KNO 降低 5.8%。

### Shallow water

最新 AM-KNO 已在 epoch 499 完成，替代了旧的部分结果。PKNO 从
2.876e-2 改善至 1.486e-2，降幅 48.3%，但 KNO 为 7.970e-3，明显更好。
这证明当前 PKNO 对 KNO 并非通用优势。浅水仍可作为“PKNO 修复 AM-KNO
大部分退化、但未恢复 KNO 水平”的限制性结果。

又由于浅水 PKNO 采用 r=4 而 KNO/AM-KNO 是 r=8，不能将该行用作干净的
组件因果归因。它适合放入附录或局限性段落。

## 复杂度快照

| 任务族 | KNO 参数量 | AM-KNO 参数量 | PKNO 参数量 | PKNO/KNO | PKNO/AM-KNO |
|---|---:|---:|---:|---:|---:|
| Burgers | 33,921 | 286,242 | 411,001 | 12.1x | 1.44x |
| 2D NS 与 SWE | 526,026 | 571,883 | 915,585 | 1.74x | 1.60x |

KNO 的 complexity.csv 记录 inference timing，而 AM-KNO/PKNO 的 metrics.csv
记录整 epoch 时间，两者不是同一计时协议。参数量可以进入论文，但必须说明
PKNO 是精度导向、参数更多的模型；暂不能发表跨实现的 runtime 排名。

## 对论文的具体修改建议

目标包：PKNOpaper/PKNO_AAAI2027_submission_package_20260730。优先修改
实际包含的主文件 PKNO_paper_framework.tex，再同步工作中的
experiment/pkno_experiment_section.tex。

| 位置 | 建议 |
|---|---|
| PKNO_paper_framework.tex，tab:pkno_main_results，约第 700 行 | 将 KNO 的 NS e-3 Step/Full 改为 1.551e-2 / 1.629e-2；NS e-4 改为 4.118e-1 / 4.672e-1；SWE 改为 7.909e-3 / 7.970e-3。NS e-3、SWE 加粗 KNO；Burgers、NS e-4 加粗 PKNO。表注增加单 seed 和 SWE PKNO r=4 的说明。 |
| experiment/pkno_experiment_section.tex，tab:pkno_main_results | 做完全相同的数值与加粗更新。 |
| experiment/pkno_experiment_section.tex，tab:pkno_amkno_results | SWE AM-KNO Full Rel. L2 从 5.149e-2 改为 2.876e-2；PKNO 降幅从 71.1% 改为 48.3%。其余三行仍为 67.7%、55.6%、17.9%。这是当前最能突出 PKNO 的表。 |
| experiment/pkno_experiment_section.tex，tab:pkno_estimated_kno_results | 改名为“实测三方法比较”；KNO 不再是 estimate。若保留 SWE，必须说明 r=4 限制；更稳妥的做法是将 SWE 移至附录。 |
| experiment/pkno_experiment_section.tex，tab:pkno_components | 更新全部数值，但仅称为粗粒度“三变体设计比较”，不能称组件消融。KNO 到 PKNO 同时改变多项结构，SWE 还有 r 不一致。主文建议仅保留 Burgers 和两个 NS。 |
| 主结果叙述与结论 | 删除“PKNO 在两个 NS 上都低于 KNO”的表述。改成：Burgers 最优；NS e-3 与 KNO 基本持平且 KNO 点估计略优；NS e-4 PKNO 点估计略优；SWE KNO 最优。AM-KNO 降幅范围由 17.9--71.1% 改为 17.9--67.7%。 |
| figureForcomplexity.pdf | 若图含旧 MSE/timing，提交前需重画。现有产物不支持跨 KNO wrapper 与 PyTorch 实现的直接 runtime 比较。 |

## 可进入论文的结论

**主文可用：** Burgers 与两个 T=40 NS 任务的实测 KNO/AM-KNO/PKNO 相对 L2；
PKNO 相比 AM-KNO 的 67.7%、55.6%、17.9% 降幅；带参数量限制说明的复杂度
分析。

**附录或局限性使用：** 浅水三方法比较、NS T=20 诊断、首 batch 频谱指标。
浅水结果具有科学价值，但会削弱普适主张且存在 r 不一致。

**当前不能使用：** 统计显著性、PKNO 普遍优于 KNO、字典和条件传播的细粒度
因果归因、AM-PKNO 主张、以及跨当前实现的 runtime 排名。

## 来源

主要数值为以下目录中 metrics.csv 最后一行：
- First_set_of_valid_data_statistics_7_31_13_33/outputs/stage0_kno_baseline/
- First_set_of_valid_data_statistics_7_31_13_33/outputs/stage1_0_am_kno/
- First_set_of_valid_data_statistics_7_31_13_33/outputs/stage3_0_param_kno/

协议和解释依据：docs/experiments/experiment_protocol.md、
docs/experiments/stage1_0_am_kno_design.md、
docs/experiments/stage3_0_pkno_design.md、
docs/experiments/stage3_0_shallow_water_stability_notes.md、以及
docs/models/stage3_0_condition_and_dictionary_design.md。
