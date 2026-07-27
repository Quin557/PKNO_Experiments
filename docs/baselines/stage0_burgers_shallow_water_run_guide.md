# Stage 0 Burgers 与 Shallow-water 运行指南

本文件用于第一阶段后续扩展实验：

```text
KoopmanLab Burgers baseline
KoopmanLab shallow-water baseline
```

当前最优先仍是 Navier-Stokes；Burgers 和 shallow-water 在 NS smoke/full baseline 稳定后再跑。

## 1. 统一阶段目录

这两个实验仍然属于第一阶段：

```bash
STAGE=stage0_kno_baseline
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"
```

输出位置：

```text
logs/stage0_kno_baseline/<run_name>.log
outputs/stage0_kno_baseline/<run_name>/
```

## 2. Burgers 数据

目标文件：

```text
$DATA_ROOT/burgers/burgers_data_R10.mat
```

仓库模板目录：

```text
data/burgers/
```

检查：

```bash
source configs/data_paths.env
ls -lh "$DATA_ROOT/$BURGERS_FILE"
```

## 3. Burgers smoke test

```bash
source configs/data_paths.env

STAGE=stage0_kno_baseline
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

export CUDA_VISIBLE_DEVICES=0
RUN=smoke_koopmanlab_burgers_gpu0

nohup python -u experiments/official_kno/train_koopmanlab_burgers.py \
  --koopmanlab-root "$KOOPMANLAB_ROOT" \
  --data-path "$DATA_ROOT/$BURGERS_FILE" \
  --run-name "$RUN" \
  --output-dir "$OUT_DIR" \
  --epochs 1 \
  --batch-size 64 \
  --sub 32 \
  --operator-size 32 \
  --modes 16 \
  --decompose 8 \
  --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &

echo $!
```

查看：

```bash
tail -f logs/stage0_kno_baseline/smoke_koopmanlab_burgers_gpu0.log
cat outputs/stage0_kno_baseline/smoke_koopmanlab_burgers_gpu0/metrics.csv
```

## 4. Burgers full run

```bash
source configs/data_paths.env

STAGE=stage0_kno_baseline
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

RUN=kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42

CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/official_kno/train_koopmanlab_burgers.py \
  --koopmanlab-root "$KOOPMANLAB_ROOT" \
  --data-path "$DATA_ROOT/$BURGERS_FILE" \
  --run-name "$RUN" \
  --output-dir "$OUT_DIR" \
  --epochs 500 \
  --batch-size 64 \
  --sub 32 \
  --operator-size 32 \
  --modes 16 \
  --decompose 8 \
  --lr 0.001 \
  --step-size 100 \
  --gamma 0.5 \
  --seed 42 \
  --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &

echo $!
```

## 5. Shallow-water 数据

目标文件：

```text
$DATA_ROOT/shallow_water/2D_rdb_NA_NA.h5
```

仓库模板目录：

```text
data/shallow_water/
```

检查：

```bash
source configs/data_paths.env
ls -lh "$DATA_ROOT/$SHALLOW_WATER_FILE"
```

确认 HDF5 结构：

```bash
python - "$DATA_ROOT/$SHALLOW_WATER_FILE" <<'PY'
import sys, h5py

with h5py.File(sys.argv[1], "r") as f:
    keys = sorted(f.keys())
    print("root key count:", len(keys))
    print("first keys:", keys[:5])
    print("last keys:", keys[-5:])
    if "data" in f:
        print("/data", f["data"].shape, f["data"].dtype)
    else:
        first = keys[0]
        print(f"{first}/data", f[f"{first}/data"].shape, f[f"{first}/data"].dtype)
PY
```

`experiments/official_kno/train_koopmanlab_shallow_water.py` 已兼容两种格式：

```text
/data                         # KoopmanLab 扁平格式
0000/data, 0001/data, ...      # PDEBench 2D_rdb_NA_NA.h5 原始 group 格式
```

PDEBench 原始样本通常是 `(T, X, Y, C)`，脚本会在内存中转换为 KNO 需要的 `(B, X, Y, T)`，不需要提前另存一个转换后的 HDF5 文件。默认 split 为 `--ntrain 900 --ntest 100`，对应 KoopmanLab shallow-water loader 的官方默认设置。

## 6. Shallow-water smoke test

```bash
source configs/data_paths.env

STAGE=stage0_kno_baseline
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

export CUDA_VISIBLE_DEVICES=0
RUN=smoke_koopmanlab_shallow_water_gpu0

nohup python -u experiments/official_kno/train_koopmanlab_shallow_water.py \
  --koopmanlab-root "$KOOPMANLAB_ROOT" \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" \
  --run-name "$RUN" \
  --output-dir "$OUT_DIR" \
  --epochs 1 \
  --batch-size 5 \
  --t-in 10 \
  --t-out 40 \
  --sub 1 \
  --ntrain 900 \
  --ntest 100 \
  --operator-size 32 \
  --modes 16 \
  --decompose 8 \
  --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &

echo $!
```

查看：

```bash
tail -f logs/stage0_kno_baseline/smoke_koopmanlab_shallow_water_gpu0.log
cat outputs/stage0_kno_baseline/smoke_koopmanlab_shallow_water_gpu0/metrics.csv
tail -n 5 outputs/stage0_kno_baseline/smoke_koopmanlab_shallow_water_gpu0/rollout_error_by_step.csv
```

## 7. Shallow-water full run

```bash
source configs/data_paths.env

STAGE=stage0_kno_baseline
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

RUN=kno_koopmanlab_shallow_water_o32_m16_r8_t40_ep500_seed42

CUDA_VISIBLE_DEVICES=1 nohup python -u experiments/official_kno/train_koopmanlab_shallow_water.py \
  --koopmanlab-root "$KOOPMANLAB_ROOT" \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" \
  --run-name "$RUN" \
  --output-dir "$OUT_DIR" \
  --epochs 500 \
  --batch-size 5 \
  --t-in 10 \
  --t-out 40 \
  --sub 1 \
  --ntrain 900 \
  --ntest 100 \
  --operator-size 32 \
  --modes 16 \
  --decompose 8 \
  --lr 0.001 \
  --step-size 100 \
  --gamma 0.5 \
  --seed 42 \
  --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &

echo $!
```

## 8. 汇总

```bash
python scripts/collect_results.py
cat results/stage0_kno_baseline/run_summary.csv
```
