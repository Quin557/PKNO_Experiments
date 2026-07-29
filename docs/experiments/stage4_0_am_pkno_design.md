# Stage4_0 AM-PKNO 实验设计

Stage4_0 的代号是：

```text
stage4_0_am_pkno
```

目标不是继续调 Stage3_0，而是新建一个 AM-PKNO 分支：

```text
src/ampkno/
experiments/stage4_0/
```

它把 Stage3_0 的 parameterized Koopman / shared dictionary 和 Stage1_0 的 AM-style frequency generation 合并，用实验先判断这条路是否值得发展为最终 PKNO。

## 1. 核心问题

Stage3_0 当前学习的是：

```text
K_k(c_n) = K0_k + DeltaK_phi(k, c_static, S(h_n))
```

这个设计能验证参数化 Koopman family，但仍有两个限制：

- `K0_k` 是截断 modes 上的 per-frequency 参数表；
- 2D 数据若直接为所有频率生成 `K(k,c)`，batch-specific 矩阵会非常大。

Stage4_0 要回答：

```text
用 AM-FNO 的 MLP + Chebyshev basis + factorization 生成 K(k,c_n)，
是否能在不截断高频的情况下改善长时间 rollout 和高频谱误差。
```

## 2. 从 AM-FNO 到 AM-PKNO

AM-FNO 的关键思想是把 Fourier kernel 看成频率的函数：

```text
R(k) = G_phi(e(k))
```

其中 `e(k)` 是正交频率基，MLP 版默认用 Chebyshev basis。Stage4_0 不生成 FNO 卷积 kernel，而是生成 KNO 的 Fourier-domain Koopman matrix：

```text
K_k(c_n) = G_phi(e(k), c_n)
```

这里：

```text
h_n      = 当前 history window
z_n      = Psi_theta(h_n)
s_n      = S_theta(h_n)
c_n      = ConditionEncoder(c_static, s_n)
e(k)     = ChebyshevFrequencyEmbedding(k)
K_k(c_n) = AM generator 输出的 complex Koopman matrix
```

单步预测：

```text
z_hat_n = FFT(Psi_theta(h_n))
z_hat_{n+1,k} = K_k(c_n) z_hat_{n,k}
u_{n+1} = D_theta(IFFT(z_hat_{n+1}))
```

## 3. 为什么共享字典不接收条件

PKNN / parametric Koopman 的核心假设是不同参数条件共享一个 observable subspace，参数只改变作用在这个 subspace 上的 Koopman operator。

因此 Stage4_0 保留 Stage3_0 的边界：

```text
z_n = Psi_theta(h_n)
```

而不是：

```text
z_n = Psi_theta(h_n, c)
```

如果把 `c` 直接拼进 dictionary，实验上就很难区分提升来自参数化 Koopman family，还是来自普通 conditional encoder。

## 4. 2D conditioned factorization

2D 全频率直接生成：

```text
K(kx, ky, c) = G_phi(e(kx, ky), c)
```

会对每个 `(kx, ky)` 频率点和每个 batch 样本生成一套 `O x O` complex matrix。对 NS/shallow-water，这会明显放大显存和时间。

Stage4_0 默认使用 AM-FNO MLP 风格的轴向分解，但让轴向因子也接收条件：

```text
K(kx, ky, c)[i,o] = sum_r Gx(kx, c)[i,o,r] * Gy(ky, c)[i,o,r]
```

默认：

```text
--operator-factorization factorized
--factorized-rank 1
--max-modes 0
```

`max_modes=0` 表示使用当前 FFT 网格可用的所有频率。若显存不足，再把它改成 `--max-modes 16` 或 `--max-modes 24`，并在 run name 中写清 `cap16/cap24`。

实现注意：

```text
K(kx,ky,c) 的 factorized 公式不能直接用三操作数 einsum 一步算完。
```

初版直接使用：

```text
einsum("bixy,bxior,byior->boxy")
```

在 NS/shallow-water 的 `t_out=40, decompose=8` full run 中会触发 OOM。原因是 PyTorch 可能隐式保存 `[B,Kx,Ky,O,O,R]` 级别中间张量，并在 autoregressive rollout 中累积数百次。当前代码已改为 chunked memory-efficient contraction，数学表达不变，但避免 materialize 完整 batch-specific 2D Koopman matrix。

默认实现参数：

```text
--factorized-input-chunk 32
```

这个参数不改变使用所有频率的事实，只是控制一次 contraction 处理多少个 observable input。你已经实测 `chunk32` 时显存最多只到约 40%，所以主线直接定为 `32`。它比 `--max-modes 16` 更适合作为主实验加速/稳显存手段，因为它保留 `max_modes=0` 的 all-frequency AM-PKNO 设定。

同时，Stage4_0 默认开启 `checkpoint_koopman=true`，对 repeated Koopman updates 使用 activation checkpoint。它会让训练变慢，但能显著降低 `t_out * decompose` 展开图的显存压力。

## 5. 四个数据集默认设计

| Dataset | Stage4_0 default | 设计理由 |
|---|---|---|
| Burgers | 1D `K(k,c_n)` all-frequency | 快速检查 AM-PKNO 的 shape、rollout 和频谱指标 |
| NS v1e-3 | 2D conditioned factorized all-frequency | 主要流体实验，粘性条件明确，保持 `t_out=40` |
| NS v1e-4 | 2D conditioned factorized all-frequency | 低粘度高频更强，默认更小 `lr/output_scale` |
| Shallow-water | 2D conditioned factorized all-frequency, conservative rollout | 沿用修复后的 HDF5 loader，降低 `decompose/lr/output_scale` 控制发散 |

特别注意：

```text
ns_V1e-4_N10000_T30.mat 当前服务器文件实际有 50 帧，
所以 Stage4_0 默认使用 t_in=10, t_out=40，并显式使用 ntrain=1000, ntest=200。
```

## 6. 第一轮对照关系

Stage4_0 的主对照不是只看它是否绝对最优，而是看趋势：

| 对照 | 判断目标 |
|---|---|
| Stage0 KNO vs Stage4_0 AM-PKNO | fixed truncated K 是否被 all-frequency conditioned generator 改善 |
| Stage1_0 AM-KNO vs Stage4_0 AM-PKNO | 加入 shared dictionary 和 physical/state condition 是否更稳 |
| Stage3_0 Param-KNO vs Stage4_0 AM-PKNO | 去掉 per-mode table、改成 AM 生成是否改善高频和长 rollout |
| NS v1e-3 vs NS v1e-4 | `log10(viscosity)` 是否能被稳定吸收 |
| Shallow-water smoke/full | factorized conditioned operator 是否比 Stage3 shallow-water 更稳定 |

## 7. 输出和指标

沿用现有训练工具输出：

```text
outputs/stage4_0_am_pkno/<run_name>/
  args.json
  config.yaml
  env.txt
  metrics.csv
  rollout_error_by_step.csv
  spectral_metrics.csv
```

重点看：

```text
test_full_rel_l2
test_step_rel_l2
rollout_error_by_step.csv
high_band_rel_l2
gradient_rel_l2
```

如果只降低单步误差，但 `test_full_rel_l2`、高频误差或 rollout slope 变差，不应认为 AM-PKNO 成功。

## 8. 后续更典型实验

四个现有数据集跑稳后，更适合突出 AM-PKNO 的后续实验是：

```text
PDEBench CFD-1D / CFD-2D
```

推荐 baseline：

```text
FNO
AM-FNO
KNO
AM-KNO
Param-KNO
AM-PKNO
```

原因是 CFD-1D/2D 更接近 AM-FNO 的优势场景，也更适合做跨物理条件、强高频、长时间 rollout 的统一比较。
