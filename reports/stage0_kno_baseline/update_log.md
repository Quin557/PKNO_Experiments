# Stage 0 KNO Update Log

更新日期：2026-07-29

## 本轮新增

- `experiments/official_kno/stage0_kno_metrics.py`
- `docs/server/stage0_kno_server_run_guide.md`

## 本轮修改

- `experiments/official_kno/train_koopmanlab_burgers.py`
- `experiments/official_kno/train_koopmanlab_ns.py`
- `experiments/official_kno/train_koopmanlab_shallow_water.py`

## 说明

1. 三个 Stage 0 训练入口现在都会在训练结束后直接调用统一评估模块。  
2. 同一个 run 目录里会自动生成 `checkpoint_last.pt`、`metrics.csv`、`rollout_error_by_step.csv`、`spectral_metrics.csv`、`complexity.csv`、`evaluation_summary.json`。  
3. Stage 0 server guide 已改成单命令训练+自动落盘的流程，默认不再要求手工补跑 evaluation-only。  
4. 本轮只处理 Stage 0 KNO baseline，没有修改 AM-KNO、Param-KNO 或 Stage 4。

## 结果路径

```text
outputs/stage0_kno_baseline/<run_name>/
logs/stage0_kno_baseline/<run_name>.log
reports/stage0_kno_baseline/checkpoint_inventory.md
```
