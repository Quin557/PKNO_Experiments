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

AM-KNO 在 Stage1_0 对应改成最纯粹的频率生成版本：

```text
K_k = G_phi(e(k))
z_hat_{n+1,k} = K_k z_hat_{n,k}
```

暂时不把当前状态放进 Koopman matrix 生成器。下面这种形式留给 Stage3 / PKNO 条件化阶段：

```text
K_k = G_phi(e(k), S_theta(history_n))
z_hat_{n+1,k} = K_k z_hat_{n,k}
```

理由是 Stage1_0 应先回答 AM 思想本身是否有价值。若同时引入 `S_theta(history_n)`，提升或退化很难区分来自 AM 频率生成，还是来自当前状态条件化。

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

## 4. 2D Factorized Generator

初版 2D AM-KNO 使用完整二维频率输入：

```text
K_(kx,ky) = G_phi(e(kx, ky))
```

这会对每个笛卡尔积频率点都跑一次大 MLP。以 `128x128` shallow-water 为例，`rfft2` 约有：

```text
128 x 65 = 8320
```

个频率点。如果还使用 `condition_mode=state`，每个 batch 都会生成：

```text
B x 8320 x O x O
```

个 complex Koopman matrix，速度和显存都会被放大。

Stage1_0 现在默认采用更贴近 AM-FNO MLP 版的 2D 分解形式：

```text
K_(kx,ky)[i,o] = sum_r Gx_phi(e(kx))[i,o,r] * Gy_phi(e(ky))[i,o,r]
```

默认：

```text
--operator-factorization factorized
--factorized-rank 1
```

`rank=1` 最接近 AM-FNO 的 x/y 方向乘积分解。若后续需要更强表达能力，可以试 `--factorized-rank 2` 或 `4`，但第一轮建议先保持 `1`。

旧完整 2D generator 仍可用于消融：

```text
--operator-factorization full
```

但不作为默认 full run。

## 5. mode 参数处理

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

## 6. 当前模型文件

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

## 7. 四个数据集默认设计

| Dataset | Default AM variant | Reason |
|---|---|---|
| Burgers | `K_k=G(e(k))` | 快速检查 1D 高频谱和 mesh 行为，避免过早引入状态条件 |
| NS v1e-3 | `K_k=G(e(k))`, 2D factorized | 先验证纯 AM 对涡量 rollout 高频结构是否有帮助 |
| NS v1e-4 | `K_k=G(e(k))`, 2D factorized | 低粘性更强调小尺度和高频稳定性，先避免状态条件干扰 |
| Shallow-water | `K_k=G(e(k))`, 2D factorized | all-frequency 下必须控制 2D generator 成本，沿用保守训练超参 |

## 8. 推荐消融

第一轮主跑：

```text
A1: condition_mode=freq, operator_factorization=factorized
A2: condition_mode=freq, operator_factorization=full
```

可选：

```text
A3: condition_mode=freq, operator_factorization=factorized + --factorized-rank 2
A4: condition_mode=freq, operator_factorization=factorized + --use-hf-residual
```

`condition_mode=state` 不作为 Stage1_0 默认消融。它应放到 Stage3 / PKNO 条件化 Koopman 实验中。

判断标准：

```text
Full Rel L2 是否下降
High-band spectral error 是否下降
Gradient Rel L2 是否下降
Rollout error growth 是否更慢
显存和训练时间是否可接受
```

如果只提升单步误差，但高频误差或长期 rollout 不稳，不能说明 AM-KNO 成功。

## 9. 速度预期

对 2D all-frequency 数据，factorized freq-only 主要减少两类成本：

```text
生成器 MLP 调用:  O(B * Nx * Ny) -> O(Nx + Ny)
batch 权重维度:   B x Nx x Ny x O x O -> Nx x Ny x O x O
```

实现上会先生成小的 x/y 轴向因子，再在每个 rollout step 合成一次频率矩阵 `K_(kx,ky)`，并在 `decompose` 的多次 Koopman iteration 中复用。这样比每次 iteration 都重新组合因子更快，也避免了旧 `state` 版本的 batch-specific 权重张量。

以 shallow-water `B=5, 128x128, O=32, rank=1` 粗略估算，generator MLP 调用会从约 `5*8320` 个频率条件降到 `128+65` 个轴向条件。但总 epoch 速度不会同步提升两百倍，因为 FFT、materialized `K_(kx,ky)` 和 Koopman time marching 仍然存在：

```text
t_out x decompose = 40 x 8
```

保守预期是 shallow-water 从约 `414-417s/epoch` 降到几十秒到两三分钟区间；如果原 run 主要受权重生成和显存带宽拖累，可能接近 `2x-5x` 加速。实际值需要服务器 smoke 复测。

## 10. 推荐后续更典型实验

若四个现有数据集看到正向趋势，建议下一步做：

```text
PDEBench CFD-1D / CFD-2D
baseline: FNO, AM-FNO, KNO, AM-KNO
```

原因是 AM-FNO 论文中 CFD-1D/2D 的高频成分更强，也更适合突出 AM 对高频信息保留的价值。
