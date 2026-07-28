# Stage 0 部分日志评估：Burgers 完成，NS 发散排查

评估对象来自服务器 `logs/stage0_kno_baseline/` 的四个日志文件：

```text
kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42.log
kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_seed42.log
kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_seed42.log
kno_koopmanlab_shallow_water_o32_m16_r8_t40_ep500_seed42.log
```

## 1. 总体结论

`Burgers` 已完成，可作为当前 Stage 0 的有效 baseline 结果。

`Shallow-water` 没跑完，但日志显示训练稳定下降；当前部分日志没有发现明显发散。

`NS v1e-3` 在第 27 个 epoch 出现突发式发散，不能作为正式 baseline。

`NS v1e-4` 本次 run 不能作为正式 baseline。服务器检查确认该文件不是 v1e-3 误放：`ns_V1e-4_N10000_T30.mat` 的 `u` shape 为 `(50, 64, 64, 10000)`，所以本次 `t_in=10, t_out=40` 在实际文件上是可行的。该 run 的问题更可能来自低粘度 NS 本身更难、`ntrain=1000` 相比 KoopmanLab 默认 `8000` 更少，以及 `lr=0.005` 长训稳定性不足。

## 2. 数值摘要

| Run | 已记录 epoch | 最优 Eval Pred MSE | 最后 Eval Pred MSE | 判断 |
|---|---:|---:|---:|---|
| Burgers | 0-499 | 2.873726e-05 | 2.873726e-05 | completed |
| NS v1e-3 | 0-412 | 4.878228e-03 at epoch 16 | 9.613168e+00 | diverged |
| NS v1e-4 | 0-404 | 1.753523e+00 at epoch 3 | 4.529644e+00 | unstable_or_undertrained_setting |
| Shallow-water | 0-232 | 7.515554e-05 at epoch 223 | 7.825019e-05 | interrupted_but_stable |

## 3. NS v1e-3 发散现象

NS v1e-3 前期是正常下降的：

```text
epoch 16 Eval Pred MSE = 0.004878228
epoch 26 Eval Pred MSE = 0.009347757
```

但第 27 个 epoch 突然跳变：

```text
epoch 27 Train Pred MSE = 5.531237
epoch 27 Eval Pred MSE  = 9.205549
```

之后长期卡在约 `9.6` 的平台。这不是服务器 kill 造成的尾部缺失，而是训练过程中模型已经发散。

当前参数确实来自 KoopmanLab 官方 demo：

```text
o=32, m=16, r=8, lr=0.005, step_size=100, gamma=0.5, batch_size=10, t_in=10, t_out=40
```

但官方 demo 默认只示例 `ep=1`，不能保证 `lr=0.005` 跑 500 epoch 一定稳定。对于正式复现实验，建议保留这次日志为 `diverged_official_lr005`，然后增加一个保守复跑：

```text
kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_lr001_seed42_rerun1
```

保守复跑建议：

```text
--lr 0.001
```

该结果应标记为 stability rerun，不直接混同为完全官方 demo 参数。

## 4. NS v1e-4 本次 run 的问题

本次日志显示：

```text
dataset shape: (50, 64, 64, 1200)
Navier-Stokes (vis = 1e-4) custom split loaded: ntrain=1000, ntest=200
X train shape: torch.Size([1000, 64, 64, 10])
Y train shape: torch.Size([1000, 64, 64, 40])
```

后续服务器检查显示完整 v1e-4 文件为：

```text
FILE: data/navier_stokes/ns_V1e-4_N10000_T30.mat
keys: ['a', 't', 'u']
u shape: (50, 64, 64, 10000)
```

因此，虽然文件名带 `T30`，但当前文件的实际时间维度是 50。Stage 0 可以继续使用 `--t-in 10 --t-out 40`。本次 v1e-4 不应再归因为“错文件”，而应归因为当前训练设置下的低粘度 NS 稳定性/样本量问题。

后续如果换数据源，仍建议先检查 shape：

```bash
source configs/data_paths.env

python - "$DATA_ROOT/$NS2D_V1E3_FILE" "$DATA_ROOT/$NS2D_V1E4_FILE" <<'PY'
import h5py
import sys
from pathlib import Path

for item in sys.argv[1:]:
    p = Path(item)
    print("FILE:", p)
    with h5py.File(p, "r") as f:
        print("keys:", list(f.keys()))
        print("u shape:", f["u"].shape)
PY
```

预期：

```text
v1e-3: time dimension = 50
v1e-4: 当前服务器文件 time dimension = 50
```

## 5. 下一步执行建议

1. Burgers：保留为 Stage 0 有效 baseline。
2. Shallow-water：GPU 恢复后可以继续完整重跑；当前趋势正常。
3. NS v1e-4：不要使用当前高 loss run 作为正式 baseline；建议先做 `lr=0.001` 稳定性复跑，并保留官方 `lr=0.005` 失败记录。
4. NS v1e-3：不要使用当前发散结果；建议先做一个 `lr=0.001` 的稳定性复跑，同时保留官方 `lr=0.005` 发散日志作为实验记录。

## 6. 仓库修正

本次已同步修正：

```text
docs/server/server_run_checklist.md
docs/data/stage0_data_download.md
configs/experiment/ns2d_rollout.yaml
experiments/official_kno/train_koopmanlab_ns.py
```

其中 `train_koopmanlab_ns.py` 会在训练前检查 NS 数据时间维度，避免 v1e-4 文件或 `t_out` 配错后继续消耗 GPU 时间。
