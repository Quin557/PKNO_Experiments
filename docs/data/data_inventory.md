# 数据清单

本文件记录数据来源、服务器放置位置、适合阶段和当前状态。

## Stage 0 数据优先级

| 优先级 | 数据 | 对齐来源 | 用途 | 状态 |
|---|---|---|---|---|
| A | KoopmanLab Burgers | KoopmanLab loader / KNO 官方数据 | 快速 baseline、mesh 测试、频谱指标验证 | 待确认下载文件 |
| A | KoopmanLab Navier-Stokes | KoopmanLab `demo_ns.py` / KNO 官方数据 | 主 KNO baseline、长 rollout、高频分析 | 待确认下载文件 |
| B | Shallow-water | KoopmanLab / KNO 论文 | 辅助 rollout baseline | 待确认来源 |
| B | Rayleigh-Benard | KNO 论文相关来源 | 中后期长 rollout | 暂缓 |
| B | PDEBench 参数化 PDE | PDEBench 官方数据 | 后续 Param-KNO / PKNO | 暂缓 |

## 服务器推荐目录

```text
$DATA_ROOT/
  burgers/
  navier_stokes/
  shallow_water/
  rayleigh_benard/
  pdebench/
```

私有路径写入：

```text
configs/data_paths.env
```

该文件不能提交到 git。

## 数据卡规则

每个数据集都需要在 `ref/data_cards/` 中建立数据卡，至少包含：

- 数据来源 URL；
- 许可/开放情况；
- 下载方式；
- 原始 shape；
- 处理后 shape；
- 物理变量；
- 物理参数；
- 边界条件；
- 适合阶段；
- 已知坑。
