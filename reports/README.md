# 报告目录

本目录用于放置阶段性实验报告和论文级汇总。所有论文数值必须能够回溯到
`results/experiment_result_inventory.csv` 的 `source` 字段。

## 当前报告

| Scope | Report | 用途 |
|---|---|---|
| 全局 | `experiment_results_report.md` | 当前全部结果、失败记录和论文使用规则 |
| Stage 0 | `stage0_kno_baseline/stage0_partial_log_evaluation_2026_07_28.md` | 早期失败与中断日志诊断 |
| Stage 0 | `stage0_kno_baseline/stage0_completed_experiment_evaluation_2026_07_29.md` | 已完成 KNO 结果分析 |
| Stage 0 | `stage0_kno_baseline/burgers_baseline_evaluation.md` | Burgers 单项详细评估 |
| Stage 1 | `stage1_0_am_kno/stage1_completed_experiment_evaluation_2026_07_29.md` | 已完成 AM-KNO 结果分析 |
| Stage 3 | `stage3_0_param_kno/stage3_completed_experiment_evaluation_2026_07_29.md` | 已完成 Param-KNO 结果及诊断分析 |
| Stage 4 | `stage4_0_am_pkno/stage4_completed_experiment_evaluation_2026_07_29.md` | 当前已完成 PKNO/Burgers 分析 |

Stage 2 高频分支已不在当前实验计划中，因此不创建 Stage 2 结果报告。

## 更新规则

1. 新实验先追加到 `results/experiment_result_inventory.csv`。
2. 完成执行、论文可用性和科学结论分别判断。
3. 失败、被替代和协议不匹配记录不得被成功复跑覆盖。
4. 阶段报告记录训练行为和局部结论；全局报告负责跨阶段比较。
5. 每个数值必须注明具体 `metrics.csv` 或日志来源。
