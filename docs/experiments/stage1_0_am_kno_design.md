# Stage1_0 AM-KNO 实验设计

Stage1_0 的代号是：

```text
stage1_0_am_kno
```

目标是先把 AM 思想作为一个独立候选模型验证，而不是提前并入 PKNO：

```text
src/amkno/
```

当前 AM-KNO 不修改 `src/pkno/models`。如果后续结果证明 AM 对 KNO 高频信息确实有帮助，再讨论是否进入 PKNO 主线。

## 1. 核心问题

Stage 0 已经说明四个 KNO baseline 能跑通：

- `nsv1e3`
- `nsv1e4`
- `burgers`
- `shallow_water`

Stage1_0 回答的问题是：

```text
把 KNO 中每个频率独立学习的 Koopman matrix
改成 AM-FNO 式的频率到 matrix 的共享生成函数，
是否能改善高频信息、梯度误差和 rollout 稳定性。
```

## 2. 从 AM-FNO 到 AM-KNO

AM-FNO 的关键不是增加更多高频参数，而是把 Fourier kernel 看成频率的函数：

```text
R(k) = NN_re(e(k)) + i NN_im(e(k))
```

AM-KNO 对应改成：

```text
K_k = G_phi(e(k))
z_hat_{n+1,k} = K_k z_hat_{n,k}
```

状态条件化版本为：

```text
K_k = G_phi(e(k), S_theta(history_n))
z_hat_{n+1,k} = K_k z_hat_{n,k}
```

这里的 `S_theta(history_n)` 只用于 AM-KNO 的当前状态摘要，不等价于 Stage3_0 / PKNO 的物理参数化 Koopman family。

## 3. 为什么默认 MLP + Chebyshev

AM-FNO 提供两类生成器：

```text
KAN frequency generator
MLP + orthogonal basis frequency generator
```

Stage1_0 默认采用 MLP + Chebyshev 频率基：

- 依赖更轻，不需要额外 KAN 包。
- AM-FNO 代码中 MLP 版本用正交频率基缓解普通 MLP 的 spectral bias。
- 第一轮实验应先减少变量，避免把 KAN 的训练时间和稳定性问题混进 AM-KNO 是否有效的问题。

KAN 可以作为后续消融，但不放进第一版默认命令。

## 4. mode 参数处理

固定 KNO 有核心超参：

```text
modes
```

因为每个 retained mode 都有一组独立可学习 Koopman matrix。

AM-KNO 不再把 `modes` 作为同等含义的模型容量超参。默认设置为：

```text
max_modes = 0
```

含义是使用当前 FFT 网格可用的全部频率。若显存或速度压力太大，可以设置：

```text
--max-modes 16
```

但这只是计算预算上限，不是“每个 mode 一套参数”的 KNO 超参。

## 5. 当前模型文件

```text
src/amkno/dictionary.py
src/amkno/frequency.py
src/amkno/operators.py
src/amkno/highfreq.py
src/amkno/models.py
```

训练入口：

```text
experiments/stage1_0/train_am_kno_burgers.py
experiments/stage1_0/train_am_kno_ns_v1e3.py
experiments/stage1_0/train_am_kno_ns_v1e4.py
experiments/stage1_0/train_am_kno_shallow_water.py
```

## 6. 四个数据集默认设计

| Dataset | Default AM variant | Reason |
|---|---|---|
| Burgers | `K_k=G(e(k))` | 快速检查 1D 高频谱和 mesh 行为，避免过早引入状态条件 |
| NS v1e-3 | `K_k=G(e(k), S(history))` | 涡量 rollout 的高频结构与当前状态统计强相关 |
| NS v1e-4 | `K_k=G(e(k), S(history))` | 低粘性更强调小尺度和高频稳定性，使用更保守 lr/output_scale |
| Shallow-water | `K_k=G(e(k), S(history))` | 冲击前沿和边界效应明显，沿用保守训练超参 |

## 7. 推荐消融

第一轮主跑：

```text
A1: condition_mode=freq
A2: condition_mode=state
```

可选：

```text
A3: condition_mode=freq  + --use-hf-residual
A4: condition_mode=state + --use-hf-residual
```

判断标准：

```text
Full Rel L2 是否下降
High-band spectral error 是否下降
Gradient Rel L2 是否下降
Rollout error growth 是否更慢
显存和训练时间是否可接受
```

如果只提升单步误差，但高频误差或长期 rollout 不稳，不能说明 AM-KNO 成功。

## 8. 推荐后续更典型实验

若四个现有数据集看到正向趋势，建议下一步做：

```text
PDEBench CFD-1D / CFD-2D
baseline: FNO, AM-FNO, KNO, AM-KNO
```

原因是 AM-FNO 论文中 CFD-1D/2D 的高频成分更强，也更适合突出 AM 对高频信息保留的价值。
