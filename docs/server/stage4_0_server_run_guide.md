# Stage4_0 AM-PKNO 服务器运行指南

本文件用于启动 Stage4_0，不修改旧的 `server_run_checklist.md`。

## 1. 阶段目录

```bash
STAGE=stage4_0_am_pkno
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
RESULT_DIR="results/$STAGE"
REPORT_DIR="reports/$STAGE"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$RESULT_DIR" "$REPORT_DIR"
```

每个 run 输出：

```text
outputs/stage4_0_am_pkno/<run_name>/
  args.json
  config.yaml
  env.txt
  metrics.csv
  rollout_error_by_step.csv
  spectral_metrics.csv
```

## 2. 环境

优先复用 Stage 0/1/3 已经能跑通的 PyTorch 环境。

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

Stage4_0 是 PyTorch-only，不需要 TensorFlow，也默认不需要 KAN。

## 3. 数据路径

```bash
source configs/data_paths.env

ls -lh "$DATA_ROOT/$BURGERS_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E3_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E4_FILE"
ls -lh "$DATA_ROOT/$SHALLOW_WATER_FILE"
```

NS v1e-4 注意：

```text
当前服务器上的 ns_V1e-4_N10000_T30.mat 实际 u shape 记录为 (50, 64, 64, 10000)，
所以 Stage4_0 默认使用 --t-in 10 --t-out 40，并显式写 --ntrain 1000 --ntest 200。
```

## 4. 关于 AM-PKNO

Stage4_0 默认模型是：

```text
Psi_theta(h_n)                       # shared dictionary，不接收条件
c_n = ConditionEncoder(c_static,S(h_n))
K_k(c_n) = G_phi(Chebyshev(k), c_n)  # conditioned AM Koopman generator
```

2D 数据默认：

```text
--operator-factorization factorized
--factorized-rank 1
--max-modes 0
```

`max_modes=0` 表示使用所有 FFT 频率。若显存不足，先加：

```text
--max-modes 16
```

并把 run name 中的 `allfreq` 改成 `cap16`。

## 5. Smoke Tests

先每个数据集跑 1 epoch，只确认数据读取、condition vector、conditioned AM generator、rollout 和写文件正常。

### Burgers

```bash
source configs/data_paths.env
STAGE=stage4_0_am_pkno
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

export CUDA_VISIBLE_DEVICES=0
RUN=smoke_ampkno_burgers_o32_allfreq_r8_seed42

nohup python -u experiments/stage4_0/train_am_pkno_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" \
  --run-name "$RUN" \
  --output-dir "$OUT_DIR" \
  --epochs 1 \
  --batch-size 64 \
  --sub 32 \
  --ntrain 1000 \
  --ntest 200 \
  --operator-size 32 \
  --decompose 8 \
  --max-modes 0 \
  --lr 1e-3 \
  --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &

echo $!
```

### NS v1e-3

```bash
source configs/data_paths.env
STAGE=stage4_0_am_pkno
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

export CUDA_VISIBLE_DEVICES=0
RUN=smoke_ampkno_ns_v1e3_o32_allfreq_fact1_r8_t40_seed42

nohup python -u experiments/stage4_0/train_am_pkno_ns_v1e3.py \
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
  --max-modes 0 \
  --operator-factorization factorized \
  --factorized-rank 1 \
  --output-scale 0.015 \
  --lr 5e-4 \
  --max-grad-norm 1.0 \
  --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &

echo $!
```

### NS v1e-4

```bash
source configs/data_paths.env
STAGE=stage4_0_am_pkno
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

export CUDA_VISIBLE_DEVICES=0
RUN=smoke_ampkno_ns_v1e4_o32_allfreq_fact1_r8_t40_seed42

nohup python -u experiments/stage4_0/train_am_pkno_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" \
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
  --max-modes 0 \
  --operator-factorization factorized \
  --factorized-rank 1 \
  --output-scale 0.01 \
  --lr 3e-4 \
  --max-grad-norm 1.0 \
  --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &

echo $!
```

### Shallow-water

```bash
source configs/data_paths.env
STAGE=stage4_0_am_pkno
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

export CUDA_VISIBLE_DEVICES=0
RUN=smoke_ampkno_shallow_water_o32_allfreq_fact1_r4_t40_seed42

nohup python -u experiments/stage4_0/train_am_pkno_shallow_water.py \
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
  --decompose 4 \
  --max-modes 0 \
  --operator-factorization factorized \
  --factorized-rank 1 \
  --output-scale 0.005 \
  --lr 5e-5 \
  --max-grad-norm 0.1 \
  --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &

echo $!
```

## 6. Full Runs

Smoke 通过后再跑 full。若 GPU 足够，可以一张卡跑一个实验。

```bash
source configs/data_paths.env
STAGE=stage4_0_am_pkno
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"
```

### Burgers full

```bash
RUN=ampkno_burgers_o32_allfreq_r8_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage4_0/train_am_pkno_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 64 --sub 32 \
  --ntrain 1000 --ntest 200 \
  --operator-size 32 --decompose 8 --max-modes 0 \
  --lr 1e-3 --seed 42 --save-checkpoint --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### NS v1e-3 full

```bash
RUN=ampkno_ns_v1e3_o32_allfreq_fact1_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=1 nohup python -u experiments/stage4_0/train_am_pkno_ns_v1e3.py \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 \
  --ntrain 1000 --ntest 200 \
  --operator-size 32 --decompose 8 --max-modes 0 \
  --operator-factorization factorized --factorized-rank 1 \
  --output-scale 0.015 --lr 5e-4 --max-grad-norm 1.0 \
  --seed 42 --save-checkpoint --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### NS v1e-4 full

```bash
RUN=ampkno_ns_v1e4_o32_allfreq_fact1_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=2 nohup python -u experiments/stage4_0/train_am_pkno_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 \
  --ntrain 1000 --ntest 200 \
  --operator-size 32 --decompose 8 --max-modes 0 \
  --operator-factorization factorized --factorized-rank 1 \
  --output-scale 0.01 --lr 3e-4 --max-grad-norm 1.0 \
  --seed 42 --save-checkpoint --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### Shallow-water full

```bash
RUN=ampkno_shallow_water_o32_allfreq_fact1_r4_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=3 nohup python -u experiments/stage4_0/train_am_pkno_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 5 --t-in 10 --t-out 40 \
  --ntrain 900 --ntest 100 --dt 0.01 \
  --operator-size 32 --decompose 4 --max-modes 0 \
  --operator-factorization factorized --factorized-rank 1 \
  --output-scale 0.005 --lr 5e-5 --max-grad-norm 0.1 \
  --seed 42 --save-checkpoint --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

## 7. 监控

```bash
watch -n 5 nvidia-smi
tail -f logs/stage4_0_am_pkno/<run_name>.log
tail -n 5 outputs/stage4_0_am_pkno/<run_name>/metrics.csv
cat outputs/stage4_0_am_pkno/<run_name>/spectral_metrics.csv
```

## 8. 汇总

```bash
python scripts/collect_results.py
cat results/stage4_0_am_pkno/run_summary.csv
cat results/run_summary.csv
```

## 9. 推荐顺序

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

如果 NS 或 shallow-water all-frequency 显存不足，第一轮不要改学习率，先保留其他参数并加：

```text
--max-modes 16
```

如果 shallow-water full 仍非有限，下一轮优先：

```text
--decompose 2
--output-scale 0.002
--lr 2e-5
```
