# Stage 0 KNO Server Run Guide

本文件是 Stage 0 KNO baseline 的正式服务器运行指南。  
当前目标只覆盖官方 KoopmanLab / KNO baseline，不涉及 AM-KNO、Param-KNO 或 Stage 4。

## 1. 阶段目录

```bash
STAGE=stage0_kno_baseline
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
RESULT_DIR="results/$STAGE"
REPORT_DIR="reports/$STAGE"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$RESULT_DIR" "$REPORT_DIR"
```

推荐再准备一个独立的 evaluation 输出根目录：

```bash
EVAL_OUT_DIR="outputs/stage0_kno_baseline_eval"
mkdir -p "$EVAL_OUT_DIR"
```

日志与结果约定：

```text
logs/stage0_kno_baseline/<run_name>.log
outputs/stage0_kno_baseline/<run_name>/
outputs/stage0_kno_baseline_eval/<eval_run_name>/
```

## 2. 仓库与环境

```bash
git clone git@github.com:Quin557/PKNO_Experiments.git
cd PKNO_Experiments
git pull origin main
```

优先复用已有 PyTorch 环境：

```bash
conda create -n pkno-exp python=3.10 -y
conda activate pkno-exp
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

检查：

```bash
python - <<'PY'
import torch, numpy, scipy, h5py
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

## 3. KoopmanLab

```bash
mkdir -p external
if [ ! -d external/KoopmanLab ]; then
  git clone https://github.com/Koopman-Laboratory/KoopmanLab external/KoopmanLab
fi
git -C external/KoopmanLab rev-parse HEAD
```

Stage 0 训练入口统一通过 `--koopmanlab-root external/KoopmanLab` 调用官方 API。

## 4. 数据与路径

复制私有路径配置：

```bash
cp configs/data_paths.example.env configs/data_paths.env
vim configs/data_paths.env
```

推荐数据结构：

```text
$DATA_ROOT/
  navier_stokes/
    ns_V1e-3_N5000_T50.mat
    ns_V1e-4_N10000_T30.mat
  burgers/
    burgers_data_R10.mat
  shallow_water/
    2D_rdb_NA_NA.h5
```

示例：

```bash
DATA_ROOT=/path/to/pkno_data
KOOPMANLAB_ROOT=external/KoopmanLab

BURGERS_FILE=burgers/burgers_data_R10.mat
NS2D_V1E3_FILE=navier_stokes/ns_V1e-3_N5000_T50.mat
NS2D_V1E4_FILE=navier_stokes/ns_V1e-4_N10000_T30.mat
SHALLOW_WATER_FILE=shallow_water/2D_rdb_NA_NA.h5
```

检查：

```bash
source configs/data_paths.env
ls -lh "$DATA_ROOT/$BURGERS_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E3_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E4_FILE"
ls -lh "$DATA_ROOT/$SHALLOW_WATER_FILE"
```

### 4.1 数据 shape 约定

当前服务器上的 v1e-4 文件：

```text
ns_V1e-4_N10000_T30.mat
u shape = (50, 64, 64, 10000)
```

所以 Stage 0 NS v1e-4 使用：

```text
--t-in 10 --t-out 40
```

不要再按旧猜测把它写成 `t_out=20`。

### 4.2 先做 checkpoint 清点

在启动任何长任务前先检查：

```bash
python scripts/stage0_checkpoint_inventory.py
cat reports/stage0_kno_baseline/checkpoint_inventory.md
```

若 inventory 显示 `loadable=no`，说明当前目录里还没有可用于 evaluation-only 的 `checkpoint_last.pt`。这时不能直接跑 checkpoint 复评估，只能先补跑训练。

## 5. Stage 0 输出规范

每个 run 需要保留：

```text
args.json
config.yaml
env.txt
metrics.csv
rollout_error_by_step.csv
checkpoint_last.pt
```

`time_error.pt`、预测张量、图像文件都不是 checkpoint，只能算辅助输出。

## 6. Smoke Test

先单卡 smoke，再 full。

### 6.1 Burgers smoke

```bash
source configs/data_paths.env
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

### 6.2 NS v1e-3 smoke

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=0

RUN=smoke_koopmanlab_ns_v1e3_gpu0
nohup python -u experiments/official_kno/train_koopmanlab_ns.py \
  --koopmanlab-root "$KOOPMANLAB_ROOT" \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" \
  --run-name "$RUN" \
  --output-dir "$OUT_DIR" \
  --epochs 1 \
  --batch-size 10 \
  --t-in 10 \
  --t-out 40 \
  --viscosity-type 1e-3 \
  --operator-size 32 \
  --modes 16 \
  --decompose 8 \
  --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &

echo $!
```

### 6.3 NS v1e-4 smoke

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=0

RUN=smoke_koopmanlab_ns_v1e4_gpu0
nohup python -u experiments/official_kno/train_koopmanlab_ns.py \
  --koopmanlab-root "$KOOPMANLAB_ROOT" \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" \
  --run-name "$RUN" \
  --output-dir "$OUT_DIR" \
  --epochs 1 \
  --batch-size 10 \
  --t-in 10 \
  --t-out 40 \
  --viscosity-type 1e-4 \
  --ntrain 1000 \
  --ntest 200 \
  --operator-size 32 \
  --modes 16 \
  --decompose 8 \
  --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &

echo $!
```

### 6.4 Shallow-water smoke

```bash
source configs/data_paths.env
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

## 7. Full Runs

### 7.1 Burgers full

```bash
RUN=kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42_return1
CUDA_VISIBLE_DEVICES=2 nohup python -u experiments/official_kno/train_koopmanlab_burgers.py \
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

### 7.2 NS v1e-3 full

旧的 `lr=0.005` 长训已在约 epoch 27 发散。正式复跑建议：

```bash
RUN=kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_lr001_seed42_rerun1_1
CUDA_VISIBLE_DEVICES=3 nohup python -u experiments/official_kno/train_koopmanlab_ns.py \
  --koopmanlab-root "$KOOPMANLAB_ROOT" \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" \
  --run-name "$RUN" \
  --output-dir "$OUT_DIR" \
  --epochs 500 \
  --batch-size 10 \
  --t-in 10 \
  --t-out 40 \
  --viscosity-type 1e-3 \
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

### 7.3 NS v1e-4 full

```bash
RUN=kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_lr001_seed42_rerun1
CUDA_VISIBLE_DEVICES=2 nohup python -u experiments/official_kno/train_koopmanlab_ns.py \
  --koopmanlab-root "$KOOPMANLAB_ROOT" \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" \
  --run-name "$RUN" \
  --output-dir "$OUT_DIR" \
  --epochs 500 \
  --batch-size 10 \
  --t-in 10 \
  --t-out 40 \
  --viscosity-type 1e-4 \
  --ntrain 1000 \
  --ntest 200 \
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

### 7.4 Shallow-water full

```bash
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

## 8. Evaluation-only

训练完成后，脚本会写出 `checkpoint_last.pt`。  
之后用独立评估入口生成完整测试集指标：

```bash
python experiments/official_kno/evaluate_koopmanlab_checkpoint.py \
  --run-dir outputs/stage0_kno_baseline/<run_name> \
  --dataset auto \
  --eval-run-name <run_name>_eval_v1 \
  --output-dir outputs/stage0_kno_baseline_eval
```

评估输出包括：

```text
args.json
config.yaml
env.txt
metrics.csv
rollout_error_by_step.csv
complexity.csv
spectral_metrics.csv
evaluation_summary.json
```

### 8.1 指标定义

- `step_rel_l2`
- `full_rollout_rel_l2`
- `rollout_error_by_step`
- `rollout_growth_slope`
- `complexity`
- `spectral_metrics` / `gradient_rel_l2`

## 9. 监控与排查

```bash
nvidia-smi
watch -n 5 nvidia-smi
ps -u "$USER" -f | grep train_koopmanlab | grep -v grep
tail -f logs/stage0_kno_baseline/<run_name>.log
cat outputs/stage0_kno_baseline/<run_name>/metrics.csv
```

若需要判断某个 GPU 上跑的是哪个脚本：

```bash
GPU=0
for pid in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits); do
  ps -o pid,ppid,user,etime,cmd -p "$pid"
  tr '\0' ' ' < /proc/$pid/cmdline
done
```

## 10. 阶段报告与更新日志

```text
reports/stage0_kno_baseline/checkpoint_inventory.md
reports/stage0_kno_baseline/stage0_kno_evaluation_report.md
reports/stage0_kno_baseline/update_log.md
reports/stage0_kno_baseline/stage0_completed_experiment_evaluation_2026_07_29.md
reports/stage0_kno_baseline/stage0_partial_log_evaluation_2026_07_28.md
```

## 11. 当前推荐顺序

```text
1. 先跑 checkpoint inventory
2. Burgers smoke
3. NS v1e-3 smoke
4. NS v1e-4 smoke
5. Shallow-water smoke
6. Burgers full
7. NS v1e-3 full
8. NS v1e-4 full
9. Shallow-water full
10. evaluation-only 复评估
```
