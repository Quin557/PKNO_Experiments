# IKNO Server Run Guide

## 1. Output layout

```bash
STAGE=ikno_baseline
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
RESULT_DIR="results/$STAGE"
REPORT_DIR="reports/$STAGE"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$RESULT_DIR" "$REPORT_DIR"
```

Every completed run writes:

```text
outputs/ikno_baseline/<run_name>/
  args.json
  config.yaml
  env.txt
  metrics.csv
  rollout_error_by_step.csv
  spectral_metrics.csv
  checkpoint_best.pt
  checkpoint_last.pt
```

## 2. Environment and data paths

Use the same PyTorch environment as the existing KNO baselines. IKNO is a local PyTorch implementation and does not need TensorFlow or a separate IKNO repository.

```bash
conda activate <your_torch_env>
pip install -r requirements.txt
source configs/data_paths.env

ls -lh "$DATA_ROOT/$BURGERS_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E3_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E4_FILE"
ls -lh "$DATA_ROOT/$SHALLOW_WATER_FILE"
```

Before full runs, confirm the checked-out code has the new baseline:

```bash
test -f src/ikno/models.py
test -f experiments/ikno/train_ikno_burgers.py
test -f configs/experiment/ikno/burgers_ikno.yaml
```

## 3. Primary settings

All four runs use the paper-aligned core model:

```text
operator_size=32
modes=16
decompose=4
koopman_power=2
inn_blocks=4
inn_hidden_dim=128
reconstruction_weight=0
```

The paper does not state `inn_blocks` or `inn_hidden_dim`; these local defaults are documented in `docs/models/ikno_model_design.md`. The four datasets retain the batch sizes and data splits from the existing Stage 0 baselines. Training uses Adam, `lr=1e-3`, 500 epochs, and StepLR halving every 100 epochs.

## 4. Smoke runs

Run these first with `--epochs 1`. They validate file paths, loader layouts, rollout, and output writing; they are not reportable results.

### Burgers

```bash
RUN=smoke_ikno_burgers_o32_m16_l4_p2_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/ikno/train_ikno_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 64 --sub 32 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 4 --koopman-power 2 \
  --inn-blocks 4 --inn-hidden-dim 128 --lr 1e-3 --seed 42 --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### Navier-Stokes v1e-3

```bash
RUN=smoke_ikno_ns_v1e3_o32_m16_l4_p2_t40_seed42
CUDA_VISIBLE_DEVICES=1 nohup python -u experiments/ikno/train_ikno_ns_v1e3.py \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 10 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 4 --koopman-power 2 \
  --inn-blocks 4 --inn-hidden-dim 128 --lr 1e-3 --seed 42 --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### Navier-Stokes v1e-4

```bash
RUN=smoke_ikno_ns_v1e4_o32_m16_l4_p2_t40_seed42
CUDA_VISIBLE_DEVICES=2 nohup python -u experiments/ikno/train_ikno_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 10 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 4 --koopman-power 2 \
  --inn-blocks 4 --inn-hidden-dim 128 --lr 1e-3 --seed 42 --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### Shallow-water

```bash
RUN=smoke_ikno_shallow_water_o32_m16_l4_p2_t40_seed42
CUDA_VISIBLE_DEVICES=3 nohup python -u experiments/ikno/train_ikno_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 5 --t-in 10 --t-out 40 --ntrain 900 --ntest 100 --dt 0.01 \
  --operator-size 32 --modes 16 --decompose 4 --koopman-power 2 \
  --inn-blocks 4 --inn-hidden-dim 128 --lr 1e-3 --seed 42 --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

## 5. Full runs

Replace `--epochs 1` with `--epochs 500 --step-size 100 --gamma 0.5 --save-checkpoint` in the corresponding smoke command. Use the following distinct run names:

```text
ikno_burgers_o32_m16_l4_p2_ep500_seed42
ikno_ns_v1e3_o32_m16_l4_p2_t40_ep500_seed42
ikno_ns_v1e4_o32_m16_l4_p2_t40_ep500_seed42
ikno_shallow_water_o32_m16_l4_p2_t40_ep500_seed42
```

Do not overwrite a smoke directory with a full run.

## 6. Monitor and accept

```bash
watch -n 5 nvidia-smi
tail -f "$LOG_DIR/<run_name>.log"
tail -n 5 "$OUT_DIR/<run_name>/metrics.csv"
cat "$OUT_DIR/<run_name>/config.yaml"
```

After completion, confirm that `metrics.csv`, `rollout_error_by_step.csv`, `spectral_metrics.csv`, `checkpoint_best.pt`, and `checkpoint_last.pt` exist. Compare `test_full_rel_l2`, the per-step rollout error, and spectral metrics with the same dataset's Stage 0 KNO run. A lower one-step error alone is not sufficient if the long rollout or high-band error is worse.
