# 实验记录规范

## 每次运行必须保存

每个正式实验应该写出：

```text
outputs/<stage_name>/<run_name>/
  config.yaml
  args.json
  env.txt
  metrics.csv
  spectral_metrics.csv
  rollout_error_by_step.csv
  checkpoint_best.pt      # git 忽略
  checkpoint_last.pt      # git 忽略
```

日志应写入：

```text
logs/<stage_name>/<run_name>.log
```

阶段目录名称统一见：

```text
docs/experiments/output_layout.md
```

如果 KoopmanLab 官方脚本不能直接写这些文件，应优先使用薄包装脚本或后处理脚本补齐记录，而不是直接改模型逻辑。

## 命名规则

建议使用稳定命名：

```text
<model>_<dataset>_<setting>_ep<epochs>_seed<seed>
```

示例：

```text
kno_burgers_r10_mesh_ep500_seed42
kno_ns2d_rollout_t10_ep500_seed42
kno_ns2d_rollout_t40_ep500_seed42
```

## Baseline 规则

KNO baseline 复现优先级：

1. 官方 KoopmanLab 命令；
2. 最小包装后的官方命令；
3. 只有在记录清楚原因后，才写本地重实现。

## 什么结果算有价值

没有人为设定的成功阈值。一个结果只要满足下面条件就是有价值的：

- 数据来源清楚；
- 切分清楚；
- 命令和环境可追踪；
- 指标计算一致；
- 失败原因被诚实记录。
