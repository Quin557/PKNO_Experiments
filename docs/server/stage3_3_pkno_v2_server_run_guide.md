# Stage3_3 PKNO_v2-A 服务器运行指南

本指南包含四项主实验的完整 smoke/full 命令。不要修改旧 `src/pkno` 或
`experiments/pkno_v1`；所有输出写入 `outputs/stage3_3_pkno_v2/`。

## 0. 准备

```bash
cd /path/to/PKNO_Experiments
source /path/to/conda.sh
conda activate pkno-exp
source configs/data_paths.env
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
STAGE=stage3_3_pkno_v2
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR"
python -m compileall -q src/pkno_v2 experiments/pkno_v2
```

如果服务器还没有 `configs/data_paths.env`，先复制示例并填写真实路径：

```bash
cp configs/data_paths.example.env configs/data_paths.env
vim configs/data_paths.env
```

确认四个变量展开为文件而不是空字符串或目录：

```bash
printf 'BURGERS=%s\n' "$DATA_ROOT/$BURGERS_FILE"
printf 'NS1E3=%s\n' "$DATA_ROOT/$NS2D_V1E3_FILE"
printf 'NS1E4=%s\n' "$DATA_ROOT/$NS2D_V1E4_FILE"
printf 'SHALLOW=%s\n' "$DATA_ROOT/$SHALLOW_WATER_FILE"
test -f "$DATA_ROOT/$BURGERS_FILE"
test -f "$DATA_ROOT/$NS2D_V1E3_FILE"
test -f "$DATA_ROOT/$NS2D_V1E4_FILE"
test -f "$DATA_ROOT/$SHALLOW_WATER_FILE"
```

每个 GPU 只运行一个任务；`modes=16` 不得改动。Smoke 只跑 1 epoch，确认日志
出现 `epoch 0000` 且没有 `Non-finite` 后，再启动对应 full 命令。

T=40 的 full run 使用渐进课程：1/5/10/20/30/40 步分别从 epoch
0/40/80/120/160/200 开始。新版训练器对潜变量和物理增量做平滑限幅；若某个
长 rollout batch 不稳定，会回退到较短 horizon，并在 `metrics.csv` 的
`fallback_batches` 中记录。不要让 T=20 和 T=40 两个进程共用同一个 log 或
output run name，否则日志会交错，无法判断哪个实验真正完成。

NS `v1e-4` 只有在当前文件的 `u` 实际包含至少 50 个时间帧时才能运行
`--t-out 40`；运行前用 `h5py` 检查 shape。若文件只有 30 帧，应单独命名为
T=20 实验，不得把它写入 T=40 的结果目录。

## A1. Burgers

Smoke（GPU 0）：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/pkno_v2/train_pkno_v2_burgers.py --data-path "$DATA_ROOT/$BURGERS_FILE" --run-name burgers_smoke --output-dir "$OUT_DIR" --epochs 1 --ntrain 1000 --ntest 200 --modes 16 --decompose 8 --seed 42 --save-checkpoint > "$LOG_DIR/burgers_smoke.log" 2>&1 &
```

Full（GPU 0）：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/pkno_v2/train_pkno_v2_burgers.py --data-path "$DATA_ROOT/$BURGERS_FILE" --run-name burgers_full_s42 --output-dir "$OUT_DIR" --epochs 500 --ntrain 1000 --ntest 200 --modes 16 --decompose 8 --seed 42 --save-checkpoint > "$LOG_DIR/burgers_full_s42.log" 2>&1 &
```

## A2. NS ν=1e-3, T=40

Smoke（GPU 1）：

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u experiments/pkno_v2/train_pkno_v2_ns_v1e3.py --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" --run-name ns_v1e3_t40_smoke --output-dir "$OUT_DIR" --epochs 1 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 --modes 16 --decompose 8 --seed 42 --save-checkpoint > "$LOG_DIR/ns_v1e3_t40_smoke.log" 2>&1 &
```

Full（GPU 1）：

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u experiments/pkno_v2/train_pkno_v2_ns_v1e3.py --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" --run-name ns_v1e3_t40_full_s42 --output-dir "$OUT_DIR" --epochs 500 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 --modes 16 --decompose 8 --seed 42 --save-checkpoint > "$LOG_DIR/ns_v1e3_t40_full_s42.log" 2>&1 &
```

## A3. NS ν=1e-4, T=40

Smoke（GPU 2）：

```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u experiments/pkno_v2/train_pkno_v2_ns_v1e4.py --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" --run-name ns_v1e4_t40_smoke --output-dir "$OUT_DIR" --epochs 1 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 --modes 16 --decompose 8 --seed 42 --save-checkpoint > "$LOG_DIR/ns_v1e4_t40_smoke.log" 2>&1 &
```

Full（GPU 2）：

```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u experiments/pkno_v2/train_pkno_v2_ns_v1e4.py --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" --run-name ns_v1e4_t40_full_s42 --output-dir "$OUT_DIR" --epochs 500 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 --modes 16 --decompose 8 --seed 42 --save-checkpoint > "$LOG_DIR/ns_v1e4_t40_full_s42.log" 2>&1 &
```

## A4. Shallow-water, T=40

Smoke（GPU 3）：

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u experiments/pkno_v2/train_pkno_v2_shallow_water.py --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" --run-name shallow_water_t40_smoke --output-dir "$OUT_DIR" --epochs 1 --t-in 10 --t-out 40 --ntrain 900 --ntest 100 --modes 16 --decompose 4 --seed 42 --save-checkpoint > "$LOG_DIR/shallow_water_t40_smoke.log" 2>&1 &
```

Full（GPU 3）：

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u experiments/pkno_v2/train_pkno_v2_shallow_water.py --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" --run-name shallow_water_t40_full_s42 --output-dir "$OUT_DIR" --epochs 500 --t-in 10 --t-out 40 --ntrain 900 --ntest 100 --modes 16 --decompose 4 --seed 42 --save-checkpoint > "$LOG_DIR/shallow_water_t40_full_s42.log" 2>&1 &
```

## 运行检查与结果

```bash
tail -f "$LOG_DIR/<run>.log"
ps -u "$USER" -f | grep train_pkno_v2 | grep -v grep
find "$OUT_DIR/<run>" -maxdepth 1 -type f -printf '%f\n'
```

正式结果读取 `metrics.csv` 的最后一行 `test_full_rel_l2`，并结合
`rollout_error_by_step.csv` 与 `spectral_metrics.json`。四项均低于旧 PKNO 的
RL2 门槛后，才进行论文结果替换评估；超过 KNO/iKNO/AM-KNO 仅作为 stretch target。

第五项联合 NS 实验暂不运行，待 A1–A4 完成后另行补充命令和设计。
