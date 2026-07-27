# 第一阶段 KNO Baseline 服务器运行指南

本指南用于当前第一阶段：**跑 KoopmanLab / KNO 官方 baseline**。

当前原则：

- 优先使用 KoopmanLab 官方 API；
- 不改 KNO 模型逻辑；
- 用薄包装脚本补齐日志和结构化输出；
- 训练使用 `nohup python -u ... > logs/<run>.log 2>&1 &`；
- 每个 run 的结构化输出放在 `outputs/<run_name>/`。

## 1. 克隆仓库

```bash
git clone git@github.com:Quin557/PKNO_Experiments.git
cd PKNO_Experiments
```

如果服务器已有仓库：

```bash
cd /path/to/PKNO_Experiments
git pull origin main
```

## 2. 准备环境

优先复用已有 PyTorch 环境。若没有：

```bash
conda create -n pkno-exp python=3.10 -y
conda activate pkno-exp

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

检查环境：

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

## 3. 准备 KoopmanLab

```bash
mkdir -p external

if [ ! -d external/KoopmanLab ]; then
  git clone https://github.com/Koopman-Laboratory/KoopmanLab external/KoopmanLab
fi

git -C external/KoopmanLab rev-parse HEAD
```

本仓库的第一阶段入口会通过 `--koopmanlab-root external/KoopmanLab` 调用官方 API。

## 4. 准备数据路径

复制私有路径配置：

```bash
cp configs/data_paths.example.env configs/data_paths.env
vim configs/data_paths.env
```

推荐服务器数据结构：

```text
$DATA_ROOT/
  navier_stokes/
    ns_V1e-3_N5000_T50.mat
    ns_V1e-4_N10000_T30.mat
  burgers/
  shallow_water/
```

示例 `configs/data_paths.env`：

```bash
DATA_ROOT=/path/to/pkno_data
KOOPMANLAB_ROOT=external/KoopmanLab

NS2D_V1E3_FILE=navier_stokes/ns_V1e-3_N5000_T50.mat
NS2D_V1E4_FILE=navier_stokes/ns_V1e-4_N10000_T30.mat
```

检查数据文件：

```bash
source configs/data_paths.env
ls -lh "$DATA_ROOT/$NS2D_V1E3_FILE"
ls -lh "$DATA_ROOT/$NS2D_V1E4_FILE"
```

如果某个文件暂时没有，就先只跑已有文件。

## 5. 目录准备

```bash
mkdir -p logs outputs results reports
```

日志统一放：

```text
logs/<run_name>.log
```

结构化输出统一放：

```text
outputs/<run_name>/
```

每个 run 预计输出：

```text
outputs/<run_name>/
  args.json
  config.yaml
  env.txt
  metrics.csv
  rollout_error_by_step.csv
  time_error.pt          # git 忽略
  figures/               # 可选图和 pred_yy.pt，git 忽略大文件
```

## 6. 单卡 Smoke Test

先只用一张卡，例如物理 GPU 0：

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs outputs
```

### Smoke: NS v1e-3

```bash
RUN=smoke_koopmanlab_ns_v1e3_gpu0

nohup python -u experiments/official_kno/train_koopmanlab_ns.py \
  --koopmanlab-root "$KOOPMANLAB_ROOT" \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" \
  --run-name "$RUN" \
  --epochs 1 \
  --batch-size 10 \
  --t-in 10 \
  --t-out 40 \
  --viscosity-type 1e-3 \
  --operator-size 32 \
  --modes 16 \
  --decompose 8 \
  --device cuda \
  > "logs/$RUN.log" 2>&1 &

echo $!
```

查看进度：

```bash
tail -f logs/smoke_koopmanlab_ns_v1e3_gpu0.log
```

查看输出：

```bash
find outputs/smoke_koopmanlab_ns_v1e3_gpu0 -maxdepth 2 -type f
cat outputs/smoke_koopmanlab_ns_v1e3_gpu0/metrics.csv
tail -n 5 outputs/smoke_koopmanlab_ns_v1e3_gpu0/rollout_error_by_step.csv
```

### Smoke: NS v1e-4

如果 v1e-4 数据已准备好：

```bash
source configs/data_paths.env
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs outputs

RUN=smoke_koopmanlab_ns_v1e4_gpu0

nohup python -u experiments/official_kno/train_koopmanlab_ns.py \
  --koopmanlab-root "$KOOPMANLAB_ROOT" \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" \
  --run-name "$RUN" \
  --epochs 1 \
  --batch-size 10 \
  --t-in 10 \
  --t-out 40 \
  --viscosity-type 1e-4 \
  --operator-size 32 \
  --modes 16 \
  --decompose 8 \
  --device cuda \
  > "logs/$RUN.log" 2>&1 &

echo $!
```

## 7. Full Run

Smoke test 通过后再跑 full run。当前只有两张 RTX A6000 空闲时，建议一张卡跑一个实验。

### GPU 0: NS v1e-3 full

```bash
source configs/data_paths.env
mkdir -p logs outputs

RUN=kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_seed42

CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/official_kno/train_koopmanlab_ns.py \
  --koopmanlab-root "$KOOPMANLAB_ROOT" \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" \
  --run-name "$RUN" \
  --epochs 500 \
  --batch-size 10 \
  --t-in 10 \
  --t-out 40 \
  --viscosity-type 1e-3 \
  --operator-size 32 \
  --modes 16 \
  --decompose 8 \
  --lr 0.005 \
  --step-size 100 \
  --gamma 0.5 \
  --seed 42 \
  --device cuda \
  > "logs/$RUN.log" 2>&1 &

echo $!
```

### GPU 1: NS v1e-4 full

```bash
source configs/data_paths.env
mkdir -p logs outputs

RUN=kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_seed42

CUDA_VISIBLE_DEVICES=1 nohup python -u experiments/official_kno/train_koopmanlab_ns.py \
  --koopmanlab-root "$KOOPMANLAB_ROOT" \
  --data-path "$DATA_ROOT/$NS2D_V1E4_FILE" \
  --run-name "$RUN" \
  --epochs 500 \
  --batch-size 10 \
  --t-in 10 \
  --t-out 40 \
  --viscosity-type 1e-4 \
  --operator-size 32 \
  --modes 16 \
  --decompose 8 \
  --lr 0.005 \
  --step-size 100 \
  --gamma 0.5 \
  --seed 42 \
  --device cuda \
  > "logs/$RUN.log" 2>&1 &

echo $!
```

## 8. 监控

查看 GPU：

```bash
nvidia-smi
watch -n 5 nvidia-smi
```

查看训练进程：

```bash
ps -u "$USER" -f | grep train_koopmanlab_ns.py | grep -v grep
```

查看日志：

```bash
tail -f logs/<run_name>.log
```

查看指标：

```bash
cat outputs/<run_name>/metrics.csv
tail -n 5 outputs/<run_name>/rollout_error_by_step.csv
```

## 9. 停止任务

```bash
ps -u "$USER" -f | grep train_koopmanlab_ns.py | grep -v grep
kill <PID>
```

## 10. 汇总结果

运行结束后生成轻量 summary：

```bash
python scripts/collect_results.py
cat results/run_summary.csv
```

只提交轻量结果：

```bash
git status --short
git add README.md docs configs ref scripts experiments src tests results reports
git commit -m "Add KNO baseline experiment scaffold"
git push origin main
```

不要提交：

```text
data/
outputs/
logs/
*.pt
*.pth
*.ckpt
*.mat
*.h5
*.hdf5
```

## 11. 当前最推荐执行顺序

```text
1. 准备 KoopmanLab 官方 NS 数据
2. 跑 NS v1e-3 smoke
3. 跑 NS v1e-4 smoke
4. 跑 NS v1e-3 full
5. 跑 NS v1e-4 full
6. 汇总 metrics / rollout_error_by_step
7. 再补 spectral_metrics
8. 再扩展 Burgers baseline
```
