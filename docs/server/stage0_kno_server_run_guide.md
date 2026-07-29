# Stage 0 KNO Server Run Guide

这份文档只管 Stage 0 KNO baseline。  
当前约定是：**一次 `nohup` 训练结束后，同一输出目录里直接生成 checkpoint 和完整评估文件**，不再要求你手工再跑一遍 evaluation 脚本。

## 1. 目录约定

```bash
STAGE=stage0_kno_baseline
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
REPORT_DIR="reports/$STAGE"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$REPORT_DIR"
```

统一输出结构：

```text
logs/stage0_kno_baseline/<run_name>.log
outputs/stage0_kno_baseline/<run_name>/
```

每个 run 完成后，目录里至少应有：

```text
args.json
config.yaml
env.txt
checkpoint_last.pt
metrics.csv
rollout_error_by_step.csv
spectral_metrics.csv
complexity.csv
evaluation_summary.json
time_error.pt   # 兼容文件，可保留
```

## 2. 环境

```bash
git clone git@github.com:Quin557/PKNO_Experiments.git
cd PKNO_Experiments
git pull origin main
```

```bash
conda create -n pkno-exp python=3.10 -y
conda activate pkno-exp
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

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

训练入口统一使用 `--koopmanlab-root external/KoopmanLab`。

## 4. 数据路径

先准备私有路径配置：

```bash
cp configs/data_paths.example.env configs/data_paths.env
vim configs/data_paths.env
source configs/data_paths.env
```

建议数据结构：

```text
$DATA_ROOT/
  burgers/
    burgers_data_R10.mat
  navier_stokes/
    ns_V1e-3_N5000_T50.mat
    ns_V1e-4_N10000_T30.mat
  shallow_water/
    2D_rdb_NA_NA.h5
```

检查文件：

```bash
ls -lh "$DATA_ROOT/$BURGERS_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E3_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E4_FILE"
ls -lh "$DATA_ROOT/$SHALLOW_WATER_FILE"
```

## 5. Stage 0 输出原则

这四个 baseline 都按同一规则跑：

1. 训练脚本负责训练。
2. 训练脚本结束后自动把完整测试集评估结果写进同一个 run 目录。
3. 不再手工补跑 `evaluate_koopmanlab_checkpoint.py` 作为默认流程。
4. 新结果一律写新目录，不覆盖旧结果。

## 6. 先清点 checkpoint

```bash
python scripts/stage0_checkpoint_inventory.py
cat reports/stage0_kno_baseline/checkpoint_inventory.md
```

如果 `loadable=no`，说明当前 run 目录里还没有可直接复评估的 checkpoint，但这不影响重新训练。

## 7. 训练命令

建议先单卡 smoke test，再跑 full test。  
下面给的是**重新跑四个 baseline** 的推荐命令，run 名都加了 `rerun2`，避免覆盖旧目录。

### 7.1 Burgers

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=0

RUN=kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42_rerun2
nohup python -u experiments/official_kno/train_koopmanlab_burgers.py \
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

### 7.2 Navier-Stokes v1e-3

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=1

RUN=kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_lr001_seed42_rerun2
nohup python -u experiments/official_kno/train_koopmanlab_ns.py \
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

### 7.3 Navier-Stokes v1e-4

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=0

RUN=kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_lr001_seed42_rerun2
nohup python -u experiments/official_kno/train_koopmanlab_ns.py \
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

### 7.4 Shallow Water

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=1

RUN=kno_koopmanlab_shallow_water_o32_m16_r8_t40_ep500_seed42_rerun2
nohup python -u experiments/official_kno/train_koopmanlab_shallow_water.py \
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

## 8. 监控

```bash
nvidia-smi
watch -n 5 nvidia-smi
ps -u "$USER" -f | grep train_koopmanlab | grep -v grep
tail -f logs/stage0_kno_baseline/<run_name>.log
cat outputs/stage0_kno_baseline/<run_name>/metrics.csv
```

如果你想知道某块 GPU 上当前跑的是哪个脚本：

```bash
GPU=0
for pid in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits); do
  ps -o pid,ppid,user,etime,cmd -p "$pid"
  tr '\0' ' ' < /proc/$pid/cmdline
done
```

## 9. 推荐顺序

```text
1. checkpoint inventory
2. Burgers
3. NS v1e-3
4. NS v1e-4
5. Shallow Water
```

## 10. 结果文件

每个 run 结束后，重点看：

- `metrics.csv`
- `rollout_error_by_step.csv`
- `spectral_metrics.csv`
- `complexity.csv`
- `evaluation_summary.json`
- `checkpoint_last.pt`

`metrics.csv` 里已经包含：

```text
run_name, model, dataset, viscosity, seed, epoch,
test_mse, step_rel_l2, full_rollout_rel_l2,
rollout_growth_slope, params, peak_memory_gb,
inference_ms_per_step, rollout_ms
```

## 11. 备选：独立复评估

如果后面你只想复查某个已经训练好的 checkpoint，再单独跑：

```bash
python experiments/official_kno/evaluate_koopmanlab_checkpoint.py \
  --run-dir outputs/stage0_kno_baseline/<run_name> \
  --dataset auto \
  --eval-run-name <run_name>_eval_v1 \
  --output-dir outputs/stage0_kno_baseline_eval
```

但正常 Stage 0 baseline 流程里，这一步已经不是必须。
