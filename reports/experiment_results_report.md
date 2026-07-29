# PKNO 三阶段实验结果报告

更新时间：2026-07-29

本报告只覆盖当前论文的三阶段实验链条：KNO、AM-KNO 和 PKNO。内部
Stage 3 `ParamKNO` 即论文中的 **PKNO**。所有数值必须回溯到
`results/experiment_result_inventory.csv` 的 `source` 字段。

## 1. 阶段报告

| 阶段 | 论文名称 | 报告职责 | 详细报告 |
|---|---|---|---|
| Stage 0 | KNO | 只验证 KNO baseline | `stage0_kno_baseline/stage0_completed_experiment_evaluation_2026_07_29.md` |
| Stage 1 | AM-KNO | 分析 AM-KNO，并与 KNO 对比 | `stage1_0_am_kno/stage1_completed_experiment_evaluation_2026_07_29.md` |
| Stage 3 | PKNO | 分析 PKNO，并与 KNO、AM-KNO 对比 | `stage3_0_param_kno/stage3_pkno_completed_experiment_evaluation_2026_07_29.md` |

报告必须区分执行完成、协议可比和科学结论。完成 500 epoch 但绝对误差很高
的运行仍是有效记录，不能写成方法成功。

## 2. 论文主结果口径

论文的主要研究问题是长期 rollout accuracy and stability，因此主表采用
scale-normalized step/full relative L2。只有使用同一 PyTorch evaluator 的
AM-KNO 与 PKNO 可以直接进入当前主表。KNO wrapper 尚未导出 matched
relative L2，因此其 MSE 只用于 baseline validation，不参与主表排名。

| Dataset | Method | Step Rel. L2 | Full Rel. L2 | PKNO improvement |
|---|---|---:|---:|---:|
| Burgers | AM-KNO | `1.600e-2` | `1.600e-2` | -- |
|  | **PKNO** | **`5.163e-3`** | **`5.163e-3`** | **67.7%** |
| NS, `nu=1e-3` | AM-KNO | `3.503e-2` | `3.672e-2` | -- |
|  | **PKNO** | **`1.570e-2`** | **`1.632e-2`** | **55.6%** |
| NS, `nu=1e-4` | AM-KNO | `5.168e-1` | `5.654e-1` | -- |
|  | **PKNO** | **`4.022e-1`** | **`4.641e-1`** | **17.9%** |

三个 matched comparison 中 PKNO 均取得最低 relative L2。加粗只标记同一
数据集、同一指标和同一评估实现下的最优值。

## 3. Stage 0：KNO baseline

| Dataset | Horizon | Mean-step MSE | 判定 |
|---|---:|---:|---|
| Burgers | 1 | `2.874e-5` | 可用 baseline |
| NS, `nu=1e-3` | 40 | `2.370e-4` | 稳定复跑，可用 baseline |
| NS, `nu=1e-4` | 40 | `6.768e-1` | 完成但为负结果 |
| Shallow water | 40 | `6.913e-5` | 可用 baseline |

Stage 0 只回答 KNO 是否完成和是否形成有效 baseline，不引入 AM-KNO 或
PKNO 的结构结论。原始 NS `nu=1e-3`, `lr=0.005` 运行发散，正式值来自
`lr=0.001` 的稳定复跑。

## 4. Stage 1：AM-KNO 对 KNO

| Dataset | KNO MSE | AM-KNO MSE | 结论 |
|---|---:|---:|---|
| Burgers | `2.874e-5` | `1.267e-4` | AM-KNO 未优于 KNO |
| NS, `nu=1e-3` | `2.370e-4` | `1.198e-3` | AM-KNO 未优于 KNO |
| NS, `nu=1e-4` | `6.768e-1` | `9.540e-1` | 两者误差均高，AM-KNO 更高 |

当前结果不支持“AM-KNO 比 KNO 好”。Stage 1 的价值是提供纯
frequency-generated operator 对照：模型可以稳定训练，但仅替换频率参数化
不足以保持 KNO 的 MSE。

## 5. Stage 3：PKNO 对 KNO 和 AM-KNO

PKNO 在三个 matched relative-L2 comparison 中均优于 AM-KNO。辅助 MSE
比较显示：Burgers 上 PKNO 接近 KNO；NS `nu=1e-3` 上与 KNO 相差约
1.2%；NS `nu=1e-4` 上比 KNO 低约 1.5%，但绝对误差仍高。

因此当前最稳妥且有利的论文结论是：condition-dependent propagation 与
shared hybrid dictionary 使 PKNO 显著改善 AM-KNO，并把 MSE 恢复到 KNO
级别；在 matched relative-L2 evaluator 下，PKNO 是当前最优模型。

不能写成“PKNO 已在所有任务、所有指标上全面超过 KNO”，因为 KNO 的
relative L2 仍缺失，shallow-water 的 AM-KNO 也尚未完成。

## 6. 暂不进入论文主表的结果

- Shallow-water：PKNO 已完成，但 AM-KNO 缺失，KNO matched relative L2
  也未导出，当前不能形成三模型公平比较。
- NS `nu=1e-4`, T=20：只用于诊断 horizon degradation，不与 T=40 混合。
- Spectral metrics：当前只覆盖第一个 evaluation batch。
- Growth slope：现有 rollout CSV 保存逐步 MSE，而不是逐步 relative L2。

“暂不进入主表”表示证据或协议尚不完整，不表示删除结果。相关数值继续保留
在清单和阶段报告中，完成匹配评估后再决定正文或附录位置。

## 7. 失败、被替代与诊断记录

| Run | 类型 | 处理 |
|---|---|---|
| Stage 0 NS `nu=1e-3`, `lr=0.005` | 发散 | 不进论文；由 `lr=0.001` 完整复跑替代 |
| Stage 0 NS `nu=1e-4`, `lr=0.005` | 高损失且未完成 | 不进论文；保留为稳定性失败证据 |
| Stage 0 shallow-water 早期副本 | 服务器中断 | 由完整 500-epoch 运行替代 |
| Stage 3 NS `nu=1e-4`, T=20 | 协议不匹配 | 仅用于预测长度诊断 |
| 所有 `smoke_*` | 执行检查 | 永不作为论文结果 |

## 8. 计算成本

| Dataset | Method | Params (M) | Mean epoch wall time (s) |
|---|---|---:|---:|
| Burgers | KNO | 0.034 | 0.179 |
| Burgers | AM-KNO | 0.286 | 0.177 |
| Burgers | PKNO | 0.411 | 0.239 |
| NS, `nu=1e-4`, T=40 | KNO | 0.526 | 56.5 |
| NS, `nu=1e-4`, T=40 | AM-KNO | 0.572 | 66.1 |
| NS, `nu=1e-4`, T=40 | PKNO | 0.916 | 91.9 |

Epoch wall time 包含训练 epoch 及随后进行的测试评估。计算成本属于 RQ3，
不能因为对 PKNO 不利而从论文中删除。

## 9. 后续更新规则

1. 新 run 先追加到 `experiment_result_inventory.csv`，不得覆盖失败记录。
2. KNO、AM-KNO、PKNO 必须匹配 split、horizon、seed 和 metric implementation。
3. 正文只加粗同一协议下的最优值，不对不可比数值做视觉强调。
4. 优先补 KNO matched relative L2、AM-KNO shallow-water 和至少 3 个 seeds。
5. 完整测试集指标可用后，再加入 error-growth 和 spectral tables。
