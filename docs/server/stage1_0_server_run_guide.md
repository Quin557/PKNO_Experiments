# Stage1_0 AM-KNO 服务器运行指南

本文件用于启动独立 AM-KNO 实验，不修改旧的 `server_run_checklist.md`。

## 1. 阶段目录

```bash
STAGE=stage1_0_am_kno
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
RESULT_DIR="results/$STAGE"
REPORT_DIR="reports/$STAGE"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$RESULT_DIR" "$REPORT_DIR"
```

每个 run 输出：

```text
outputs/stage1_0_am_kno/<run_name>/
  args.json
  config.yaml
  env.txt
  metrics.csv
  rollout_error_by_step.csv
  spectral_metrics.csv
```

## 2. 环境

优先复用 Stage 0 已经能跑通 KNO 的 PyTorch 环境。

```bash
conda activate <your_torch_env>
pip install -r requirements.txt
```

检查：

```bash
python - <<'PY'
import torch, numpy, scipy, h5py, yaml
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("visible gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

Stage1_0 是 PyTorch-only，不需要 TensorFlow，也默认不需要 KAN。

## 3. 数据路径

```bash
source configs/data_paths.env

ls -lh "$DATA_ROOT/$BURGERS_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E3_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E4_FILE"
ls -lh "$DATA_ROOT/$SHALLOW_WATER_FILE"
```

## 4. 关于 `--max-modes`

AM-KNO 默认：

```text
--max-modes 0
```

含义是使用当前 FFT 网格可用的全部频率。它不同于 KNO 的 `--modes`：AM-KNO 没有给每个 mode 单独存一套 Koopman matrix，而是用共享 MLP 从频率基生成 matrix。

如果 full run 显存不够，再设置：

```text
--max-modes 16
```

这只是计算预算上限。

## 5. 关于 2D factorized generator

Stage1_0 默认使用最纯粹的 AM-KNO：

```text
K_k = G(e(k))
```

不再默认使用：

```text
K_k = G(e(k), S(history))
```

当前状态条件化留给 Stage3 / PKNO。对 2D 数据，默认进一步采用 AM-FNO MLP 风格的 x/y 方向分解：

```text
--operator-factorization factorized
--factorized-rank 1
```

它表示：

```text
K_(kx,ky)[i,o] = Gx(kx)[i,o] * Gy(ky)[i,o]
```

如果要复现旧的完整二维生成器，显式使用：

```text
--operator-factorization full
```

旧 smoke 中 shallow-water 的 `state + full 2D allfreq` 约为 `414-417s/epoch`，不建议继续作为默认 full run。先重跑本指南中的 `allfreq_fact1` smoke，再决定是否跑 500 epoch。

## 6. Smoke Tests

先每个数据集跑 1 epoch，确认数据读取、AM matrix 生成、autoregressive rollout 和频谱指标写文件正常。

### Burgers

```bash
source configs/data_paths.env
STAGE=stage1_0_am_kno
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

export CUDA_VISIBLE_DEVICES=0
RUN=smoke_amkno_burgers_o32_allfreq_r8_seed42

nohup python -u experiments/stage1_0/train_am_kno_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" \
  --run-name "$RUN" \
  --output-dir "$OUT_DIR" \
  --epochs 1 \
  --batch-size 64 \
  --sub 32 \
  --operator-size 32 \
  --decompose 8 \
  --max-modes 0 \
  --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &

echo $!
```

### NS v1e-3

```bash
source configs/data_paths.env
STAGE=stage1_0_am_kno
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

export CUDA_VISIBLE_DEVICES=0
RUN=smoke_amkno_ns_v1e3_o32_allfreq_fact1_r8_t40_seed42

nohup python -u experiments/stage1_0/train_am_kno_ns_v1e3.py \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" \
  --run-name "$RUN" \
  --output-dir "$OUT_DIR" \
  --epochs 1 \
  --batch-size 10 \
  --t-in 10 \
  --t-out 40 \
  --ntrain 1000 \
  --ntest 200 \
  --operator-size 32 \
  --decompose 8 \
  --condition-mode freq \
  --max-modes 0 \
  --operator-factorization factorized \
  --factorized-rank 1 \
  --max-grad-norm 1.0 \
  --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &

echo $!
```

### NS v1e-4

```bash
source configs/data_paths.env
STAGE=stage1_0_am_kno
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

export CUDA_VISIBLE_DEVICES=0
RUN=smoke_amkno_ns_v1e4_o32_allfreq_fact1_r8_t20_seed42

nohup python -u experiments/stage1_0/train_am_kno_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" \
  --run-name "$RUN" \
  --output-dir "$OUT_DIR" \
  --epochs 1 \
  --batch-size 10 \
  --t-in 10 \
  --t-out 20 \
  --ntrain 1000 \
  --ntest 200 \
  --operator-size 32 \
  --decompose 8 \
  --condition-mode freq \
  --max-modes 0 \
  --operator-factorization factorized \
  --factorized-rank 1 \
  --max-grad-norm 1.0 \
  --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &

echo $!
```

### Shallow-water

```bash
source configs/data_paths.env
STAGE=stage1_0_am_kno
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

export CUDA_VISIBLE_DEVICES=0
RUN=smoke_amkno_shallow_water_o32_allfreq_fact1_r8_t40_seed42

nohup python -u experiments/stage1_0/train_am_kno_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" \
  --run-name "$RUN" \
  --output-dir "$OUT_DIR" \
  --epochs 1 \
  --batch-size 5 \
  --t-in 10 \
  --t-out 40 \
  --ntrain 900 \
  --ntest 100 \
  --dt 0.01 \
  --operator-size 32 \
  --decompose 8 \
  --condition-mode freq \
  --max-modes 0 \
  --operator-factorization factorized \
  --factorized-rank 1 \
  --output-scale 0.015 \
  --max-grad-norm 0.5 \
  --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &

echo $!
```

## 7. Full Runs

Smoke 通过后再跑 full。若 GPU 足够，可以一张卡跑一个实验。

```bash
source configs/data_paths.env
STAGE=stage1_0_am_kno
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"
```

### Burgers full

```bash
RUN=amkno_burgers_o32_allfreq_r8_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage1_0/train_am_kno_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 64 --sub 32 \
  --operator-size 32 --decompose 8 --max-modes 0 \
  --lr 1e-3 --seed 42 --save-checkpoint --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### NS v1e-3 full

```bash
RUN=amkno_ns_v1e3_o32_allfreq_fact1_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=1 nohup python -u experiments/stage1_0/train_am_kno_ns_v1e3.py \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 \
  --ntrain 1000 --ntest 200 \
  --operator-size 32 --decompose 8 --condition-mode freq --max-modes 0 \
  --operator-factorization factorized --factorized-rank 1 \
  --lr 5e-4 --output-scale 0.03 --max-grad-norm 1.0 \
  --seed 42 --save-checkpoint --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### NS v1e-4 full

```bash
RUN=amkno_ns_v1e4_o32_allfreq_fact1_r8_t20_ep500_seed42
CUDA_VISIBLE_DEVICES=2 nohup python -u experiments/stage1_0/train_am_kno_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 20 \
  --ntrain 1000 --ntest 200 \
  --operator-size 32 --decompose 8 --condition-mode freq --max-modes 0 \
  --operator-factorization factorized --factorized-rank 1 \
  --lr 3e-4 --output-scale 0.02 --max-grad-norm 1.0 \
  --seed 42 --save-checkpoint --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### Shallow-water full

```bash
RUN=amkno_shallow_water_o32_allfreq_fact1_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=3 nohup python -u experiments/stage1_0/train_am_kno_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 5 --t-in 10 --t-out 40 \
  --ntrain 900 --ntest 100 --dt 0.01 \
  --operator-size 32 --decompose 8 --condition-mode freq --max-modes 0 \
  --operator-factorization factorized --factorized-rank 1 \
  --lr 2e-4 --output-scale 0.015 --max-grad-norm 0.5 \
  --seed 42 --save-checkpoint --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

## 8. 监控

```bash
watch -n 5 nvidia-smi
tail -f logs/stage1_0_am_kno/<run_name>.log
tail -n 5 outputs/stage1_0_am_kno/<run_name>/metrics.csv
cat outputs/stage1_0_am_kno/<run_name>/spectral_metrics.csv
```

## 9. 汇总

```bash
python scripts/collect_results.py
cat results/stage1_0_am_kno/run_summary.csv
cat results/run_summary.csv
```

## 10. 推荐顺序

```text
1. Burgers smoke
2. NS v1e-4 smoke
3. NS v1e-3 smoke
4. Shallow-water smoke
5. Burgers full
6. NS v1e-4 full
7. NS v1e-3 full
8. Shallow-water full
```

若任一 full run 显存压力过大，先重跑同一配置但加：

```text
--max-modes 16
```

并在 run name 中把 `allfreq` 改成 `cap16`。
