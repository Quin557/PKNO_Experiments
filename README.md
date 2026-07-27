# PKNO Experiments / PKNO 实验仓库

## 中文说明

本仓库是一个长期维护的 PKNO / KNO-family 实验工作区。它不是一个简单的阶段性复现仓库，而是用于管理 KNO baseline、后续模型优化、消融实验、参数化 Koopman 实验以及最终 PKNO 实验结果的大型实验仓库。

当前最重要的任务是：

1. 严格优先对齐 **KoopmanLab 官方仓库**。
2. 其次对齐 **KNO 论文设置**。
3. 使用 KNO 官方/论文实际使用或说明的数据。
4. 先完成可信的 KNO baseline，再做后续优化实验。

当前阶段：

```text
Stage 0: 数据管线 + KNO 官方 baseline
```

后续路线：

```text
Stage 1: 频率条件化 / AM-style KNO 高频增强
Stage 2: KNO 小结构消融
Stage 3: Parameterized K(u_n, c) 与 shared dictionary 原型
Stage 4: PKNO + 条件化 Koopman 融合
Stage 5: 高维复杂系统长时间 rollout 展示
```

第一阶段数据优先级：

1. KoopmanLab Burgers：快速 KNO baseline、mesh 测试、频谱指标验证。
2. KoopmanLab Navier-Stokes：官方 KNO baseline、长期 rollout、高频分析。
3. KoopmanLab/KNO 论文中可获取的 shallow-water 等辅助数据。
4. PDEBench 参数化数据：后续 Param-KNO / PKNO 阶段使用。

文档入口与分类规则见：

[docs/README.md](docs/README.md)

仓库结构：

```text
configs/
  data_paths.example.env
  model/
  experiment/

docs/
  README.md
  overview/
  data/
  baselines/
  experiments/
  metrics/
  models/
  server/
  reports/

ref/
  papers/
  code_notes/
  data_cards/
  equations/
  figures/

external/
  KoopmanLab/
  pknn_reference/
  PDEBench/

data/
  raw/
  processed/
  index/

src/pkno/
experiments/
scripts/
tests/
outputs/
results/
reports/
```

第一步执行目标：

```text
克隆 KoopmanLab -> 准备 KNO 官方数据 -> 跑 smoke baseline -> 记录 metrics/env/config -> 汇总结果
```

服务器操作见：

[docs/server/server_run_checklist.md](docs/server/server_run_checklist.md)

服务器输出目录规范见：

[docs/experiments/output_layout.md](docs/experiments/output_layout.md)

第一阶段数据下载与放置说明见：

[docs/data/stage0_data_download.md](docs/data/stage0_data_download.md)

Burgers 与 shallow-water 后续运行指南见：

[docs/baselines/stage0_burgers_shallow_water_run_guide.md](docs/baselines/stage0_burgers_shallow_water_run_guide.md)

总体任务顺序和研究思路见：

[docs/overview/task_order_and_research_logic.md](docs/overview/task_order_and_research_logic.md)

提交规则：

可以提交：

- 源代码；
- 配置模板；
- 中文实验文档；
- `ref/` 下的小型结构化说明；
- 轻量 CSV 结果摘要；
- 最终报告和筛选后的图。

不要提交：

- 原始数据；
- checkpoint；
- 大日志；
- 私有路径配置 `configs/data_paths.env`；
- `.mat`、`.hdf5`、`.h5`、`.npy`、`.npz`、`.pt`、`.pth`、`.ckpt`。

## English Overview

This repository is a long-term experiment workspace for PKNO and KNO-family neural operator research. It is designed to manage official KNO baselines, later model improvements, ablation studies, parameterized Koopman experiments, and final PKNO results.

The current priority is:

1. Align with the official **KoopmanLab** repository first.
2. Align with the **KNO paper** second.
3. Use the datasets actually used or specified by the official KNO code/paper.
4. Establish reliable KNO baselines before implementing later PKNO variants.

Active stage:

```text
Stage 0: Data pipeline + official KNO baselines
```

Roadmap:

```text
Stage 1: Frequency-conditioned / AM-style KNO high-frequency enhancement
Stage 2: KNO structure ablations
Stage 3: Parameterized K(u_n, c) and shared dictionary prototypes
Stage 4: PKNO + conditioned Koopman fusion
Stage 5: High-dimensional long-rollout demonstrations
```

First execution target:

```text
Clone KoopmanLab -> prepare official KNO data -> run smoke baseline -> record metrics/env/config -> summarize results
```

See the server guide:

[docs/server/server_run_checklist.md](docs/server/server_run_checklist.md)

See the server output layout:

[docs/experiments/output_layout.md](docs/experiments/output_layout.md)

See Stage 0 data download and placement:

[docs/data/stage0_data_download.md](docs/data/stage0_data_download.md)

See the overall task order and research logic:

[docs/overview/task_order_and_research_logic.md](docs/overview/task_order_and_research_logic.md)
