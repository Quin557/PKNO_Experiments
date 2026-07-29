# Stage 0 KNO Update Log

更新时间：2026-07-29

## 本轮新增

- `experiments/official_kno/checkpoint_utils.py`
- `scripts/stage0_checkpoint_inventory.py`
- `experiments/official_kno/evaluate_koopmanlab_checkpoint.py`
- `docs/server/stage0_kno_server_run_guide.md`
- `reports/stage0_kno_baseline/checkpoint_inventory.md`
- `reports/stage0_kno_baseline/stage0_kno_evaluation_report.md`
- `reports/stage0_kno_baseline/update_log.md`

## 本轮修改

- `experiments/official_kno/train_koopmanlab_burgers.py`
- `experiments/official_kno/train_koopmanlab_ns.py`
- `experiments/official_kno/train_koopmanlab_shallow_water.py`

## 说明

1. 三个 Stage 0 训练入口现在会在训练完成后保存 `checkpoint_last.pt`。
2. checkpoint 必须包含 `model_state_dict` 和 `optimizer_state_dict`。
3. 新增独立 evaluation-only 入口，后续追加指标时不需要重新训练。
4. 新增 Stage 0 专用 server guide，完整命令已从总 checklist 拆分出去。
5. 当前 Stage 0 outputs 中还没有可加载 checkpoint，因此本轮只完成检查与工具链准备，没有启动长时间重训。

## 结果可追溯路径

```text
reports/stage0_kno_baseline/checkpoint_inventory.md
reports/stage0_kno_baseline/stage0_kno_evaluation_report.md
outputs/stage0_kno_baseline/<run_name>/
logs/stage0_kno_baseline/<run_name>.log
```
