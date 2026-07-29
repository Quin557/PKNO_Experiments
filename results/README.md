# 结果摘要目录

本目录只放可以提交到 git 的轻量结果摘要。

推荐文件：

```text
experiment_result_inventory.csv
run_summary.csv
kno_official_baseline_summary.csv
spectral_metric_summary.csv
ablation_summary.csv
complexity_summary.csv
```

`experiment_result_inventory.csv` 是论文结果的主清单。新增实验时追加记录，
并保留失败、被替代和协议不匹配的 run；不要只保留最佳结果。

不要把原始 `outputs/`、checkpoint 或大日志放到这里。
