# Stage3_2 PKNO_v1 Server Run Guide

This guide contains the complete commands for PKNO_v1 smoke runs, all required
full runs and ablations, and the fifth joint NS comparison. It never changes
`modes=16`.

## 1. Setup and data checks

```bash
cd /path/to/PKNO_Experiments
source /path/to/conda.sh
conda activate pkno-exp
source configs/data_paths.env
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

STAGE=stage3_2_pkno_v1
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

python - <<'PY'
import h5py, os, scipy.io, torch
root = os.environ['DATA_ROOT']
paths = {
    'ns_v1e3': os.path.join(root, os.environ['NS2D_V1E3_FILE']),
    'ns_v1e4': os.path.join(root, os.environ['NS2D_V1E4_FILE']),
    'shallow': os.path.join(root, os.environ['SHALLOW_WATER_FILE']),
    'burgers': os.path.join(root, os.environ['BURGERS_FILE']),
}
for name, path in paths.items():
    assert os.path.isfile(path), path
    print(name, path, os.path.getsize(path))
for name in ('ns_v1e3', 'ns_v1e4'):
    with h5py.File(paths[name]) as f:
        print(name, 'u shape=', f['u'].shape)
assert scipy.io.loadmat(paths['burgers'])['a'].shape[0] >= 1200
print('cuda=', torch.cuda.is_available(), 'gpus=', torch.cuda.device_count())
PY

pytest -q tests/test_pkno_v1_model.py tests/test_pkno_v1_stability.py \
  tests/test_pkno_v1_loaders.py tests/test_pkno_v1_rollout_smoke.py
```

Expected NS shape is `[50, 64, 64, N]`; the `v1e-4` filename may say `T30`,
but the current server file must expose 50 stored frames for `T_in=10,T_out=40`.

## 2. Four smoke tests

Run these after setup. Each command is fully independent.

```bash
RUN=smoke_pknov1_burgers_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/pkno_v1/train_pkno_v1_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 64 --operator-size 32 --modes 16 --decompose 8 --lr 1e-3 \
  --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &

RUN=smoke_pknov1_ns_v1e3_seed42
CUDA_VISIBLE_DEVICES=1 nohup python -u experiments/pkno_v1/train_pkno_v1_ns_v1e3.py \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 10 --t-in 10 --t-out 40 --operator-size 32 --modes 16 --decompose 8 \
  --lr 5e-4 --max-grad-norm 1.0 --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &

RUN=smoke_pknov1_ns_v1e4_seed42
CUDA_VISIBLE_DEVICES=2 nohup python -u experiments/pkno_v1/train_pkno_v1_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 10 --t-in 10 --t-out 40 --operator-size 32 --modes 16 --decompose 8 \
  --lr 5e-4 --max-grad-norm 1.0 --growth-weight 1e-3 --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &

RUN=smoke_pknov1_shallow_seed42
CUDA_VISIBLE_DEVICES=3 nohup python -u experiments/pkno_v1/train_pkno_v1_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 5 --t-in 10 --t-out 40 --operator-size 32 --modes 16 --decompose 4 \
  --lr 5e-5 --delta-scale 0.005 --max-grad-norm 0.1 --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &
```

Check all four before any full run:

```bash
for f in "$LOG_DIR"/smoke_pknov1_*.log; do echo "===== $f"; tail -n 12 "$f"; done
find "$OUT_DIR" -maxdepth 2 -name evaluation_summary.json -o -name metrics.csv
nvidia-smi
```

## 3. Validation-only tuning runs

These two commands establish the Burgers and shallow-water settings without
reading their historical test trajectories. `evaluation_summary.json` is
intentionally absent from these runs because `--skip-test-evaluation` is set.

```bash
RUN=tune_pknov1_burgers_900train_100val_seed42
CUDA_VISIBLE_DEVICES=4 nohup python -u experiments/pkno_v1/train_pkno_v1_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --split-mode tuning --skip-test-evaluation --epochs 500 --batch-size 64 --operator-size 32 --modes 16 --decompose 8 \
  --lr 1e-3 --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &

RUN=tune_pknov1_shallow_800train_100val_seed42
CUDA_VISIBLE_DEVICES=5 nohup python -u experiments/pkno_v1/train_pkno_v1_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --split-mode tuning --skip-test-evaluation --epochs 500 --batch-size 5 --t-in 10 --t-out 40 \
  --operator-size 32 --modes 16 --decompose 4 --lr 5e-5 --delta-scale 0.005 --max-grad-norm 0.1 \
  --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &
```

Inspect validation rather than test metrics:

```bash
tail -n 8 "$OUT_DIR/tune_pknov1_burgers_900train_100val_seed42/metrics.csv"
tail -n 8 "$OUT_DIR/tune_pknov1_shallow_800train_100val_seed42/metrics.csv"
```

## 4. Four primary full runs

These are the only four runs eligible to replace Stage3_0 PKNO. Keep the
commands exactly at seed 42 and modes 16.

```bash
RUN=pknov1_burgers_o32_m16_r8_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/pkno_v1/train_pkno_v1_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --split-mode final --epochs 500 --batch-size 64 --operator-size 32 --modes 16 --decompose 8 \
  --lr 1e-3 --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &

RUN=pknov1_ns_v1e3_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=1 nohup python -u experiments/pkno_v1/train_pkno_v1_ns_v1e3.py \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 --operator-size 32 --modes 16 --decompose 8 \
  --lr 5e-4 --max-grad-norm 1.0 --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &

RUN=pknov1_ns_v1e4_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=2 nohup python -u experiments/pkno_v1/train_pkno_v1_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 --operator-size 32 --modes 16 --decompose 8 \
  --lr 5e-4 --max-grad-norm 1.0 --growth-weight 1e-3 --state-weight 1e-4 --smooth-weight 1e-4 \
  --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &

RUN=pknov1_shallow_o32_m16_r4_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=3 nohup python -u experiments/pkno_v1/train_pkno_v1_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --split-mode final --epochs 500 --batch-size 5 --t-in 10 --t-out 40 --operator-size 32 --modes 16 --decompose 4 \
  --lr 5e-5 --delta-scale 0.005 --max-grad-norm 0.1 --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &
```

## 5. Required full ablations

Run these after the primary runs. They identify which V1 change drives each
dataset result.

```bash
RUN=ablation_pknov1_burgers_direct_o32_m16_r8_ep500_seed42
CUDA_VISIBLE_DEVICES=4 nohup python -u experiments/pkno_v1/train_pkno_v1_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --split-mode final --epochs 500 --batch-size 64 --operator-size 32 --modes 16 --decompose 8 \
  --lr 1e-3 --direct-prediction --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &

RUN=ablation_pknov1_ns_v1e3_physics_only_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=5 nohup python -u experiments/pkno_v1/train_pkno_v1_ns_v1e3.py \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 --operator-size 32 --modes 16 --decompose 8 \
  --lr 5e-4 --max-grad-norm 1.0 --gate-max 0 --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &

RUN=ablation_pknov1_ns_v1e4_no_growth_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=6 nohup python -u experiments/pkno_v1/train_pkno_v1_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 --operator-size 32 --modes 16 --decompose 8 \
  --lr 5e-4 --max-grad-norm 1.0 --growth-weight 0 --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &

RUN=ablation_pknov1_shallow_no_curriculum_o32_m16_r4_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=7 nohup python -u experiments/pkno_v1/train_pkno_v1_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --split-mode final --epochs 500 --batch-size 5 --t-in 10 --t-out 40 --operator-size 32 --modes 16 --decompose 4 \
  --lr 5e-5 --delta-scale 0.005 --max-grad-norm 0.1 --one-step-epochs 0 --short-rollout-epochs 0 \
  --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &
```

## 6. Promotion check

After the four primary runs finish, execute this exact check. Do not replace
the paper result if any line says `FAIL`.

```bash
OUT_DIR=outputs/stage3_2_pkno_v1 python - <<'PY'
import json, os
checks = {
    'pknov1_burgers_o32_m16_r8_ep500_seed42': 5.163e-3,
    'pknov1_ns_v1e3_o32_m16_r8_t40_ep500_seed42': 1.632e-2,
    'pknov1_ns_v1e4_o32_m16_r8_t40_ep500_seed42': 4.641e-1,
    'pknov1_shallow_o32_m16_r4_t40_ep500_seed42': 1.486e-2,
}
root = os.environ['OUT_DIR']; passed = True
for run, threshold in checks.items():
    path = os.path.join(root, run, 'evaluation_summary.json')
    value = json.load(open(path))['full_rel_l2']
    ok = value < threshold
    print(run, f'{value:.8e}', '<', f'{threshold:.8e}', 'PASS' if ok else 'FAIL')
    passed &= ok
raise SystemExit(0 if passed else 1)
PY
```

## 7. Fifth experiment: joint NS

Run this section only after the promotion check passes. All models train on the
same balanced `1000 + 1000` trajectories and test on the same balanced
`200 + 200` trajectories. KNO/iKNO/AM-KNO/original PKNO commands use one
script; its validation set is never the final test set.

```bash
RUN=joint_ns_kno_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/pkno_v1/train_ns_joint_baseline.py \
  --model kno --data-v1e3 "$DATA_ROOT/$NS2D_V1E3_FILE" --data-v1e4 "$DATA_ROOT/$NS2D_V1E4_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR/joint_baselines" --epochs 500 --batch-size 10 --t-in 10 --t-out 40 \
  --operator-size 32 --modes 16 --decompose 8 --lr 1e-3 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &

RUN=joint_ns_ikno_o32_m16_l4p2_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=1 nohup python -u experiments/pkno_v1/train_ns_joint_baseline.py \
  --model ikno --data-v1e3 "$DATA_ROOT/$NS2D_V1E3_FILE" --data-v1e4 "$DATA_ROOT/$NS2D_V1E4_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR/joint_baselines" --epochs 500 --batch-size 10 --t-in 10 --t-out 40 \
  --operator-size 32 --modes 16 --decompose 8 --lr 1e-3 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &

RUN=joint_ns_amkno_o32_allfreq_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=2 nohup python -u experiments/pkno_v1/train_ns_joint_baseline.py \
  --model amkno --data-v1e3 "$DATA_ROOT/$NS2D_V1E3_FILE" --data-v1e4 "$DATA_ROOT/$NS2D_V1E4_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR/joint_baselines" --epochs 500 --batch-size 10 --t-in 10 --t-out 40 \
  --operator-size 32 --modes 16 --decompose 8 --lr 5e-4 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &

RUN=joint_ns_pkno_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=3 nohup python -u experiments/pkno_v1/train_ns_joint_baseline.py \
  --model pkno --data-v1e3 "$DATA_ROOT/$NS2D_V1E3_FILE" --data-v1e4 "$DATA_ROOT/$NS2D_V1E4_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR/joint_baselines" --epochs 500 --batch-size 10 --t-in 10 --t-out 40 \
  --operator-size 32 --modes 16 --decompose 8 --lr 5e-4 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &

RUN=joint_ns_pknov1_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=4 nohup python -u experiments/pkno_v1/train_pkno_v1_ns_joint.py \
  --data-v1e3 "$DATA_ROOT/$NS2D_V1E3_FILE" --data-v1e4 "$DATA_ROOT/$NS2D_V1E4_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR" --epochs 500 --batch-size 10 --t-in 10 --t-out 40 \
  --operator-size 32 --modes 16 --decompose 8 --lr 5e-4 --growth-weight 1e-3 \
  --seed 42 --device cuda --save-checkpoint > "$LOG_DIR/$RUN.log" 2>&1 &
```

## 8. Monitoring, resume, and OOM handling

```bash
watch -n 5 nvidia-smi
tail -f "$LOG_DIR/pknov1_ns_v1e4_o32_m16_r8_t40_ep500_seed42.log"
ps -u "$USER" -f | grep -E 'train_pkno_v1|train_ns_joint_baseline' | grep -v grep
```

Resume a stopped PKNO_v1 run from its saved optimizer/model state:

```bash
RUN=pknov1_ns_v1e4_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=2 nohup python -u experiments/pkno_v1/train_pkno_v1_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 --operator-size 32 --modes 16 --decompose 8 \
  --lr 5e-4 --max-grad-norm 1.0 --growth-weight 1e-3 --seed 42 --device cuda --save-checkpoint \
  --resume "$OUT_DIR/$RUN/checkpoint_last.pt" > "$LOG_DIR/$RUN.resume.log" 2>&1 &
```

If a V1 run exceeds memory, first reduce only batch size: NS `10 -> 5`,
shallow-water `5 -> 3`, Burgers `64 -> 32`. Do not change `modes=16`; do not
raise `decompose`; retain NS/Burgers `decompose=8` and shallow `decompose=4`.

## 9. Required output files

Every PKNO_v1 run writes:

```text
args.json
config.yaml
env.txt
metrics.csv
stability_config.json
checkpoint_final.pt
checkpoint_last.pt                  # when --save-checkpoint is set
checkpoint_best_val.pt              # only when validation exists
evaluation_summary.json
rollout_error_by_step.csv
spectral_metrics.csv
```

Do not commit data, checkpoints, outputs, or logs. Commit only code, tests,
configs, docs, and result reports after results have been reviewed.
