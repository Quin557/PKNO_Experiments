# Stage3_1 PKNO-U 服务器运行指南

本指南只服务 `stage3_1_param_kno_u`，不修改 `server_run_checklist.md`。
它完整列出四个数据集上 A/B/C 三种条件模型的 smoke 与 full 命令。一次只启动一个
同 GPU 的任务；不要把同一数据集的 A/B/C 同时塞进同一张 GPU。

```text
A = physical_only
B = physical_compact_state
C = physical_gated_state
```

## 1. 环境与代码检查

```bash
cd /path/to/PKNO_Experiments
conda activate <your_torch_env>
pip install -r requirements.txt pytest

python - <<'PY'
import torch, numpy, scipy, h5py, yaml, pytest
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

python -m pytest -q \
  tests/test_stage3_1_pkno_u_shapes.py \
  tests/test_stage3_1_unet_frequency.py \
  tests/test_stage3_1_conditioning.py \
  tests/test_stage3_1_stability.py \
  tests/test_stage3_1_rollout_smoke.py
```

不要在未通过测试时启动 full run。

## 2. 数据与目录

每次新终端开始时先执行：

```bash
source configs/data_paths.env

ls -lh "$DATA_ROOT/$BURGERS_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E3_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E4_FILE"
ls -lh "$DATA_ROOT/$SHALLOW_WATER_FILE"

STAGE=stage3_1_param_kno_u
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
RESULT_DIR="results/$STAGE"
REPORT_DIR="reports/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "$RESULT_DIR" "$REPORT_DIR"
```

每个 run 输出：

```text
outputs/stage3_1_param_kno_u/<run_name>/
  args.json
  config.yaml
  env.txt
  metrics.csv
  rollout_error_by_step.csv
  spectral_metrics.csv
  stability_diagnostics.csv
  checkpoint_best.pt       # 仅 full 的 --save-checkpoint 时生成，git 忽略
  checkpoint_last.pt       # 仅 full 的 --save-checkpoint 时生成，git 忽略
```

## 3. Smoke Tests

执行顺序必须为 A -> B -> C。仅当前一个 smoke 的 `metrics.csv` 无 NaN/Inf、
`matrix_spectral_max <= 0.98` 且显存可接受时，才启动下一个。

### 3.1 Burgers smoke

#### A

```bash
RUN=smoke_pknou_a_burgers_o32_m16_r8_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 64 --sub 32 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 \
  --condition-mode physical_only --max-operator-norm 0.98 \
  --lr 1e-3 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### B

```bash
RUN=smoke_pknou_b_burgers_o32_m16_r8_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 64 --sub 32 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 \
  --condition-mode physical_compact_state --max-operator-norm 0.98 \
  --lr 1e-3 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### C

```bash
RUN=smoke_pknou_c_burgers_o32_m16_r8_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 64 --sub 32 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 \
  --condition-mode physical_gated_state --max-operator-norm 0.98 \
  --lr 1e-3 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### 3.2 NS v1e-3 smoke

#### A

```bash
RUN=smoke_pknou_a_ns_v1e3_o32_m16_r8_t40_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_ns_v1e3.py \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 10 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 \
  --condition-mode physical_only --max-operator-norm 0.98 \
  --lr 5e-4 --max-grad-norm 1.0 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### B

```bash
RUN=smoke_pknou_b_ns_v1e3_o32_m16_r8_t40_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_ns_v1e3.py \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 10 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 \
  --condition-mode physical_compact_state --max-operator-norm 0.98 \
  --lr 5e-4 --max-grad-norm 1.0 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### C

```bash
RUN=smoke_pknou_c_ns_v1e3_o32_m16_r8_t40_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_ns_v1e3.py \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 10 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 \
  --condition-mode physical_gated_state --max-operator-norm 0.98 \
  --lr 5e-4 --max-grad-norm 1.0 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### 3.3 NS v1e-4 smoke

#### A

```bash
RUN=smoke_pknou_a_ns_v1e4_o32_m16_r8_t40_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 10 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 \
  --condition-mode physical_only --max-operator-norm 0.98 \
  --lr 3e-4 --max-grad-norm 1.0 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### B

```bash
RUN=smoke_pknou_b_ns_v1e4_o32_m16_r8_t40_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 10 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 \
  --condition-mode physical_compact_state --max-operator-norm 0.98 \
  --lr 3e-4 --max-grad-norm 1.0 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### C

```bash
RUN=smoke_pknou_c_ns_v1e4_o32_m16_r8_t40_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 10 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 \
  --condition-mode physical_gated_state --max-operator-norm 0.98 \
  --lr 3e-4 --max-grad-norm 1.0 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### 3.4 Shallow-water smoke

Shallow-water 的 A/B/C 首轮全部固定 `decompose=2`，不得直接改为 `r8`。

#### A

```bash
RUN=smoke_pknou_a_shallow_water_o32_m16_r2_t40_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 5 --t-in 10 --t-out 40 --ntrain 900 --ntest 100 --dt 0.01 \
  --operator-size 32 --modes 16 --decompose 2 \
  --condition-mode physical_only --max-operator-norm 0.98 \
  --delta-scale 0.005 --hf-residual-scale 0.02 \
  --lr 5e-5 --max-grad-norm 0.1 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### B

```bash
RUN=smoke_pknou_b_shallow_water_o32_m16_r2_t40_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 5 --t-in 10 --t-out 40 --ntrain 900 --ntest 100 --dt 0.01 \
  --operator-size 32 --modes 16 --decompose 2 \
  --condition-mode physical_compact_state --max-operator-norm 0.98 \
  --delta-scale 0.005 --hf-residual-scale 0.02 \
  --lr 5e-5 --max-grad-norm 0.1 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### C

```bash
RUN=smoke_pknou_c_shallow_water_o32_m16_r2_t40_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 1 --batch-size 5 --t-in 10 --t-out 40 --ntrain 900 --ntest 100 --dt 0.01 \
  --operator-size 32 --modes 16 --decompose 2 \
  --condition-mode physical_gated_state --max-operator-norm 0.98 \
  --delta-scale 0.005 --hf-residual-scale 0.02 \
  --lr 5e-5 --max-grad-norm 0.1 --seed 42 --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

## 4. 验收与监控

```bash
watch -n 5 nvidia-smi
tail -f "$LOG_DIR/<run_name>.log"
tail -n 5 "$OUT_DIR/<run_name>/metrics.csv"
cat "$OUT_DIR/<run_name>/stability_diagnostics.csv"
```

Smoke 必须同时满足：

```text
metrics.csv 无 NaN/Inf
stability_diagnostics.csv 的 matrix_spectral_max <= --max-operator-norm
输出文件完整
峰值显存留有安全余量
```

若出现非有限值，停止任务并保留日志；不要只降低 learning rate 后覆盖同一个 run。
若显存不足，依次降低 batch size、`unet-base-channels`、`decompose`，每次只改一项。

## 5. Full Runs

只有对应 smoke 通过，才启动同一数据集、同一变体的 full run。下面所有 full 使用与
对应 smoke 完全相同的模型超参数和数据切分，只加入 `--epochs 500 --save-checkpoint`。

### 5.1 Burgers full

#### A

```bash
RUN=pknou_a_burgers_o32_m16_r8_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 64 --sub 32 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 --condition-mode physical_only \
  --max-operator-norm 0.98 --lr 1e-3 --seed 42 --save-checkpoint --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### B

```bash
RUN=pknou_b_burgers_o32_m16_r8_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 64 --sub 32 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 --condition-mode physical_compact_state \
  --max-operator-norm 0.98 --lr 1e-3 --seed 42 --save-checkpoint --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### C

```bash
RUN=pknou_c_burgers_o32_m16_r8_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 64 --sub 32 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 --condition-mode physical_gated_state \
  --max-operator-norm 0.98 --lr 1e-3 --seed 42 --save-checkpoint --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### 5.2 NS v1e-3 full

#### A

```bash
RUN=pknou_a_ns_v1e3_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_ns_v1e3.py \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 --condition-mode physical_only \
  --max-operator-norm 0.98 --lr 5e-4 --max-grad-norm 1.0 --seed 42 --save-checkpoint --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### B

```bash
RUN=pknou_b_ns_v1e3_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_ns_v1e3.py \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 --condition-mode physical_compact_state \
  --max-operator-norm 0.98 --lr 5e-4 --max-grad-norm 1.0 --seed 42 --save-checkpoint --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### C

```bash
RUN=pknou_c_ns_v1e3_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_ns_v1e3.py \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 --condition-mode physical_gated_state \
  --max-operator-norm 0.98 --lr 5e-4 --max-grad-norm 1.0 --seed 42 --save-checkpoint --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### 5.3 NS v1e-4 full

#### A

```bash
RUN=pknou_a_ns_v1e4_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 --condition-mode physical_only \
  --max-operator-norm 0.98 --lr 3e-4 --max-grad-norm 1.0 --seed 42 --save-checkpoint --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### B

```bash
RUN=pknou_b_ns_v1e4_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 --condition-mode physical_compact_state \
  --max-operator-norm 0.98 --lr 3e-4 --max-grad-norm 1.0 --seed 42 --save-checkpoint --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### C

```bash
RUN=pknou_c_ns_v1e4_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_ns_v1e4.py \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 \
  --operator-size 32 --modes 16 --decompose 8 --condition-mode physical_gated_state \
  --max-operator-norm 0.98 --lr 3e-4 --max-grad-norm 1.0 --seed 42 --save-checkpoint --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

### 5.4 Shallow-water full

Shallow-water 必须先完成对应 A/B/C 的 r2 smoke。r2 full 稳定前，禁止把任何变体改为 r4/r8。

#### A

```bash
RUN=pknou_a_shallow_water_o32_m16_r2_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 5 --t-in 10 --t-out 40 --ntrain 900 --ntest 100 --dt 0.01 \
  --operator-size 32 --modes 16 --decompose 2 --condition-mode physical_only \
  --max-operator-norm 0.98 --delta-scale 0.005 --hf-residual-scale 0.02 \
  --lr 5e-5 --max-grad-norm 0.1 --seed 42 --save-checkpoint --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### B

```bash
RUN=pknou_b_shallow_water_o32_m16_r2_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 5 --t-in 10 --t-out 40 --ntrain 900 --ntest 100 --dt 0.01 \
  --operator-size 32 --modes 16 --decompose 2 --condition-mode physical_compact_state \
  --max-operator-norm 0.98 --delta-scale 0.005 --hf-residual-scale 0.02 \
  --lr 5e-5 --max-grad-norm 0.1 --seed 42 --save-checkpoint --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

#### C

```bash
RUN=pknou_c_shallow_water_o32_m16_r2_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage3_1/train_pkno_u_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 5 --t-in 10 --t-out 40 --ntrain 900 --ntest 100 --dt 0.01 \
  --operator-size 32 --modes 16 --decompose 2 --condition-mode physical_gated_state \
  --max-operator-norm 0.98 --delta-scale 0.005 --hf-residual-scale 0.02 \
  --lr 5e-5 --max-grad-norm 0.1 --seed 42 --save-checkpoint --device cuda > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
```

## 6. 结果登记

运行结束后：

```bash
python scripts/collect_results.py
```

再向 `results/experiment_result_inventory.csv` 追加该 run 的命令、配置、状态和来源。
源码、文档、配置和轻量 CSV 可以在本地提交；数据、logs、outputs 与 checkpoint 不提交。
本阶段不会自动执行 `git push`。
