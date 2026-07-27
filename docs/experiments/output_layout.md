# 服务器输出目录规范

服务器上的实验输出必须按研究阶段分文件夹管理，不能把所有日志平铺在一个 `logs/` 目录下。

## 阶段目录

统一使用下面这些阶段名：

```text
stage0_kno_baseline/
stage1_highfreq_rollout_diagnostics/
stage2_structure_ablation/
stage3_conditioned_koopman/
stage4_shared_dictionary_param_kno/
stage5_pkno_fusion/
```

对应路线：

```text
官方 KNO baseline
  -> 高频/rollout 诊断
  -> 小结构消融
  -> 条件化 Koopman
  -> shared dictionary / parameterized Koopman
  -> PKNO 最终融合
```

## 日志目录

```text
logs/
  stage0_kno_baseline/
    <run_name>.log
  stage1_highfreq_rollout_diagnostics/
    <run_name>.log
  stage2_structure_ablation/
    <run_name>.log
  stage3_conditioned_koopman/
    <run_name>.log
  stage4_shared_dictionary_param_kno/
    <run_name>.log
  stage5_pkno_fusion/
    <run_name>.log
```

## 输出目录

```text
outputs/
  stage0_kno_baseline/
    <run_name>/
      args.json
      config.yaml
      env.txt
      metrics.csv
      rollout_error_by_step.csv
      spectral_metrics.csv
```

后续阶段同理：

```text
outputs/stage1_highfreq_rollout_diagnostics/<run_name>/
outputs/stage2_structure_ablation/<run_name>/
outputs/stage3_conditioned_koopman/<run_name>/
outputs/stage4_shared_dictionary_param_kno/<run_name>/
outputs/stage5_pkno_fusion/<run_name>/
```

## 结果摘要目录

```text
results/
  stage0_kno_baseline/run_summary.csv
  stage1_highfreq_rollout_diagnostics/run_summary.csv
  stage2_structure_ablation/ablation_summary.csv
  stage3_conditioned_koopman/run_summary.csv
  stage4_shared_dictionary_param_kno/run_summary.csv
  stage5_pkno_fusion/final_summary.csv
```

## 报告目录

```text
reports/
  stage0_kno_baseline/
  stage1_highfreq_rollout_diagnostics/
  stage2_structure_ablation/
  stage3_conditioned_koopman/
  stage4_shared_dictionary_param_kno/
  stage5_pkno_fusion/
```

## 第一阶段固定写法

第一阶段所有命令都应使用：

```bash
STAGE=stage0_kno_baseline
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"
```

然后：

```bash
nohup python -u ... --output-dir "$OUT_DIR" > "$LOG_DIR/$RUN.log" 2>&1 &
```
