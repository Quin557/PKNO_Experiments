# 项目 Brief

## 项目目标

本项目是一个长期维护的 PKNO 实验仓库，用于管理：

- KNO 官方 baseline；
- 后续 KNO 结构优化；
- 高频增强实验；
- 消融实验；
- 参数化 Koopman / shared dictionary 原型；
- 最终 PKNO 实验。

当前不是直接写完整 PKNO 架构，而是先完成可信的 KNO baseline 和实验管理体系。

## 当前阶段

```text
Stage 0: 数据管线 + KNO 官方 baseline
```

对齐顺序：

1. KoopmanLab 官方仓库；
2. KNO 论文；
3. 官方仓库或论文说明的数据；
4. 本地 PyTorch 重写版本。

## 当前阶段不做什么

- 不从自定义 PKNO 架构开始；
- 不用无关对比脚本作为 KNO baseline 入口；
- 不先处理大型真实数据；
- 不先做大规模超参搜索。

## 阶段路线

Stage 0：

- 克隆并阅读 KoopmanLab；
- 整理 KNO 论文和相关文献；
- 准备 Burgers / Navier-Stokes 官方数据；
- 跑 smoke test；
- 跑 full baseline；
- 输出 `metrics.csv`、`spectral_metrics.csv`、`rollout_error_by_step.csv`、`config.yaml`、`env.txt`。

Stage 1：

- 在 Stage 0 可靠后，实现频率条件化 Koopman 变体。

Stage 2：

- 做 residual Koopman、FFN adapter、高频 residual / mini U-Net 等消融。

Stage 3：

- 做 shared dictionary 与 parameterized Koopman 原型。

Stage 4：

- 融合 PKNO 与条件化 Koopman generation。

Stage 5：

- 做高维复杂系统长 rollout 展示实验。
