# Stage3_3 PKNO_v2-A 服务器运行指南

本指南包含四项主实验的完整 smoke/full 命令。不要修改旧 `src/pkno` 或
`experiments/pkno_v1`；所有输出写入 `outputs/stage3_3_pkno_v2/`。

## 0. 准备

```bash
cd /path/to/PKNO_Experiments
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
mkdir -p logs/stage3_3_pkno_v2 outputs/stage3_3_pkno_v2
python -m compileall -q src/pkno_v2 experiments/pkno_v2
```

将下面四个路径替换为服务器上的真实数据文件：

```bash
BURGERS=/data/PKNO/Burgers_data_R10.mat
NS1E3=/data/PKNO/ns_V1e-3_N10000_T30.h5
NS1E4=/data/PKNO/ns_V1e-4_N10000_T30.h5
SHALLOW=/data/PKNO/shallow-water.h5
```

每个 GPU 只运行一个任务；`modes=16` 不得改动。Smoke 只跑 1 epoch，确认日志
出现 `epoch 0000` 且没有 `Non-finite` 后，再启动对应 full 命令。

## A1. Burgers

Smoke（GPU 0）：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/pkno_v2/train_pkno_v2_burgers.py --data-path "$BURGERS" --run-name burgers_smoke --output-dir outputs/stage3_3_pkno_v2 --epochs 1 --ntrain 1000 --ntest 200 --modes 16 --decompose 8 --seed 42 --save-checkpoint > logs/stage3_3_pkno_v2/burgers_smoke.log 2>&1 &
```

Full（GPU 0）：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/pkno_v2/train_pkno_v2_burgers.py --data-path "$BURGERS" --run-name burgers_full_s42 --output-dir outputs/stage3_3_pkno_v2 --epochs 500 --ntrain 1000 --ntest 200 --modes 16 --decompose 8 --seed 42 --save-checkpoint > logs/stage3_3_pkno_v2/burgers_full_s42.log 2>&1 &
```

## A2. NS ν=1e-3, T=40

Smoke（GPU 1）：

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u experiments/pkno_v2/train_pkno_v2_ns_v1e3.py --data-path "$NS1E3" --run-name ns_v1e3_t40_smoke --output-dir outputs/stage3_3_pkno_v2 --epochs 1 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 --modes 16 --decompose 8 --seed 42 --save-checkpoint > logs/stage3_3_pkno_v2/ns_v1e3_t40_smoke.log 2>&1 &
```

Full（GPU 1）：

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u experiments/pkno_v2/train_pkno_v2_ns_v1e3.py --data-path "$NS1E3" --run-name ns_v1e3_t40_full_s42 --output-dir outputs/stage3_3_pkno_v2 --epochs 500 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 --modes 16 --decompose 8 --seed 42 --save-checkpoint > logs/stage3_3_pkno_v2/ns_v1e3_t40_full_s42.log 2>&1 &
```

## A3. NS ν=1e-4, T=40

Smoke（GPU 2）：

```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u experiments/pkno_v2/train_pkno_v2_ns_v1e4.py --data-path "$NS1E4" --run-name ns_v1e4_t40_smoke --output-dir outputs/stage3_3_pkno_v2 --epochs 1 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 --modes 16 --decompose 8 --seed 42 --save-checkpoint > logs/stage3_3_pkno_v2/ns_v1e4_t40_smoke.log 2>&1 &
```

Full（GPU 2）：

```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u experiments/pkno_v2/train_pkno_v2_ns_v1e4.py --data-path "$NS1E4" --run-name ns_v1e4_t40_full_s42 --output-dir outputs/stage3_3_pkno_v2 --epochs 500 --t-in 10 --t-out 40 --ntrain 1000 --ntest 200 --modes 16 --decompose 8 --seed 42 --save-checkpoint > logs/stage3_3_pkno_v2/ns_v1e4_t40_full_s42.log 2>&1 &
```

## A4. Shallow-water, T=40

Smoke（GPU 3）：

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u experiments/pkno_v2/train_pkno_v2_shallow_water.py --data-path "$SHALLOW" --run-name shallow_water_t40_smoke --output-dir outputs/stage3_3_pkno_v2 --epochs 1 --t-in 10 --t-out 40 --ntrain 900 --ntest 100 --modes 16 --decompose 4 --seed 42 --save-checkpoint > logs/stage3_3_pkno_v2/shallow_water_t40_smoke.log 2>&1 &
```

Full（GPU 3）：

```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u experiments/pkno_v2/train_pkno_v2_shallow_water.py --data-path "$SHALLOW" --run-name shallow_water_t40_full_s42 --output-dir outputs/stage3_3_pkno_v2 --epochs 500 --t-in 10 --t-out 40 --ntrain 900 --ntest 100 --modes 16 --decompose 4 --seed 42 --save-checkpoint > logs/stage3_3_pkno_v2/shallow_water_t40_full_s42.log 2>&1 &
```

## 运行检查与结果

```bash
tail -f logs/stage3_3_pkno_v2/<run>.log
ps -u "$USER" -f | grep train_pkno_v2 | grep -v grep
find outputs/stage3_3_pkno_v2/<run> -maxdepth 1 -type f -printf '%f\n'
```

正式结果读取 `metrics.csv` 的最后一行 `test_full_rel_l2`，并结合
`rollout_error_by_step.csv` 与 `spectral_metrics.json`。四项均低于旧 PKNO 的
RL2 门槛后，才进行论文结果替换评估；超过 KNO/iKNO/AM-KNO 仅作为 stretch target。

第五项联合 NS 实验暂不运行，待 A1–A4 完成后另行补充命令和设计。
