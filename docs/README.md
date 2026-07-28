# 文档索引

本目录按文档用途分类，后续新增 Markdown 文件时优先放到对应子目录，避免继续平铺在 `docs/` 根目录。

## 分类目录

| 目录 | 用途 | 当前文档 |
|---|---|---|
| `overview/` | 项目目标、阶段路线、研究顺序 | [project_brief.md](overview/project_brief.md), [task_order_and_research_logic.md](overview/task_order_and_research_logic.md) |
| `data/` | 数据来源、下载、放置、数据清单 | [data_inventory.md](data/data_inventory.md), [stage0_data_download.md](data/stage0_data_downvload.md) |
| `baselines/` | 官方 baseline、阶段性 baseline 运行指南 | [stage0_burgers_shallow_water_run_guide.md](baselines/stage0_burgers_shallow_water_run_guide.vmd) |
| `experiments/` | 实验记录规范、输出目录、命名规则 | [experiment_protocol.md](experiments/experiment_protocol.md), [output_layout.md](experiments/output_layout.md) |
| `metrics/` | 指标定义、评估方式、诊断说明 | [high_frequency_metrics.md](metrics/high_frequency_metrics.md) |
| `models/` | 模型设计、迁移 notes、算法决策记录 | [model_design_decisions.md](models/model_design_decisions.md), [pytorch_porting_notes_pknn.md](models/pytorch_porting_notes_pknn.md), [stage4_0_am_pkno_model_design.md](models/stage4_0_am_pkno_model_design.md) |
| `server/` | 服务器环境、训练执行、运维检查清单 | [server_run_checklist.md](server/server_run_checklist.md) |
| `reports/` | 阶段报告、准备报告、长文归档 | [project_preparation_report.md](reports/project_preparation_report.md), [pkno_experiment_preparation_report.md](reports/pkno_experiment_preparation_report.md) |

## 新增规则

- 新文档不要直接放在 `docs/` 根目录，除非它是全局索引或分类规则。
- 和某个研究阶段强绑定的运行说明，优先放入 `baselines/` 或 `experiments/`，文件名保留阶段前缀，如 `stage1_*`。
- 数据下载、路径、数据卡状态放入 `data/`；具体数据集卡片仍优先放到 `ref/data_cards/`。
- 模型想法、设计取舍、迁移计划放入 `models/`；已经形成实验方案后再链接到 `experiments/`。
- 长篇总结和历史准备材料放入 `reports/`，避免干扰日常执行文档。
