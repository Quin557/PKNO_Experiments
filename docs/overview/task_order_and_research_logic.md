# 任务顺序与研究思路

这个文件回答两个问题：

1. 当前项目总体按什么顺序推进？
2. 每个阶段为什么这样安排？

## 总体原则

本项目不是直接从最终 PKNO 架构开始，而是从实验事实出发：

```text
官方 KNO baseline
  -> 高频/rollout 诊断
  -> 小结构消融
  -> 条件化 Koopman
  -> shared dictionary / parameterized Koopman
  -> PKNO 最终融合
```

先跑真实 baseline，再根据误差、频谱、rollout 稳定性决定改什么。

## 当前任务顺序

### 0. 仓库与参考资料整理

目标：

- 仓库结构清晰；
- `ref/` 中有论文、代码说明、数据卡和公式说明；
- 数据和输出不误提交；
- 服务器操作有固定规范。

状态：已初步完成。

主要位置：

- `README.md`
- `docs/overview/project_brief.md`
- `docs/data/data_inventory.md`
- `ref/`

### 1. KNO 官方 baseline

目标：

- 优先对齐 KoopmanLab 官方仓库；
- 其次对齐 KNO 论文；
- 使用 KNO 官方/论文说明的数据；
- 跑出可追踪的 baseline 结果。

第一批建议：

1. KoopmanLab Navier-Stokes smoke test；
2. KoopmanLab Navier-Stokes full run；
3. KoopmanLab Burgers baseline；
4. 如果官方数据可得，再加 shallow-water。

必须输出：

```text
metrics.csv
rollout_error_by_step.csv
spectral_metrics.csv      # 后续补
config.yaml
args.json
env.txt
logs/<run_name>.log
```

详细操作见：

```text
docs/server/server_run_checklist.md
docs/experiments/output_layout.md
docs/baselines/stage0_burgers_shallow_water_run_guide.md
```

### 2. 指标与诊断补齐

目标：

- 不只看 MSE；
- 补充 relative L2、rollout error by step、频谱误差、gradient error；
- 判断 KNO baseline 的真正弱点。

主要位置：

```text
docs/metrics/high_frequency_metrics.md
src/pkno/metrics/spectral.py
```

### 3. 频率条件化 Koopman 实验

目标：

测试共享频率生成是否比固定频率 Koopman matrix 更好。

候选：

```text
K_k = G(e(k))
K_k = G(e(k), u_embed)
```

只在 baseline 和指标稳定后开始。

### 4. 小结构消融

目标：

筛选哪些结构真的值得进入最终模型。

候选：

- residual linear Koopman；
- decoder-side FFN；
- post-Koopman FFN；
- output high-frequency residual；
- mini U-Net refinement。

判断标准：

- 长 rollout 是否更稳；
- 高频误差是否下降；
- 跨分辨率是否不崩；
- 复杂度是否值得。

### 5. Parameterized-KNO

目标：

引入物理条件、边界条件、当前状态摘要等条件信息：

```text
K_k = G(e(k), c)
K_k = G(e(k), u_embed, c)
```

参考：

```text
external/pknn_reference
ref/code_notes/pknn_reference_code_map.md
docs/models/pytorch_porting_notes_pknn.md
```

### 6. PKNO 最终融合

目标：

把 shared dictionary、parameterized Koopman family、frequency-conditioned generation 和可选高频 residual 合并。

形式暂定：

```text
z_t = Psi_theta(u_t, x, c)
K_k = G_phi(e(k), u_embed, c, bc)
z_{t+1,k} = K_k z_{t,k}
u_pred = D_theta(z_{t+1})
u_final = u_pred + optional_high_frequency_residual
```

这不是当前阶段要实现的内容。

## 当前你应该看哪里

总体路线：

```text
docs/overview/task_order_and_research_logic.md
docs/overview/project_brief.md
```

数据清单：

```text
docs/data/data_inventory.md
docs/data/stage0_data_download.md
ref/data_cards/
```

第一阶段服务器操作：

```text
docs/server/server_run_checklist.md
docs/experiments/output_layout.md
```

实验记录规范：

```text
docs/experiments/experiment_protocol.md
```

指标说明：

```text
docs/metrics/high_frequency_metrics.md
```
