# Stage4_0 AM-PKNO 模型设计说明

Stage4_0 模型代号：

```text
AM-PKNO
stage4_0_am_pkno
```

代码位置：

```text
src/ampkno/
```

训练入口：

```text
experiments/stage4_0/
```

本阶段不是修改 Stage3_0 Param-KNO，而是新增一个独立模型族，用来验证：

```text
shared dictionary + parameterized Koopman family + AM-style frequency generation
```

是否能改善 KNO-family 在高频信息和长时间 rollout 上的表现。

## 1. 设计来源

AM-PKNO 合并三条已经在项目中分开整理过的思想：

| 来源 | 项目对应 | AM-PKNO 中的角色 |
|---|---|---|
| KNO / KoopmanLab | `external/KoopmanLab/koopmanlab/models/kno.py` | 在 Fourier 域对 observable field 做 Koopman time marching |
| PKNN / Parametric Koopman | `external/pknn_reference/`, `src/pkno/models/param_kno.py` | 使用 parameter-independent shared dictionary，并让 Koopman operator 随条件变化 |
| AM-FNO | `external/NeurIPS-2024-amortized-fourier-neural-operators-Supplemental-Conference/` | 用 MLP + Chebyshev basis 把频率映射到 complex kernel/operator，并用 2D factorization 控制成本 |

## 2. 核心公式

输入 history：

```text
h_n = [u_{n-t_in+1}, ..., u_n]
```

共享字典：

```text
z_n = Psi_theta(h_n)
```

动态条件：

```text
s_n = S_theta(h_n)
c_n = C_theta(c_static, s_n)
```

AM-style Koopman 生成：

```text
e_k = ChebyshevFrequencyEmbedding(k)
K_k(c_n) = G_phi(e_k, c_n)
```

Fourier Koopman 推进：

```text
z_hat_n = FFT(z_n)
z_hat_{n+1,k} = K_k(c_n) z_hat_{n,k}
u_{n+1} = D_theta(IFFT(z_hat_{n+1}))
```

rollout 时将 `u_{n+1}` 追加回 history window，继续下一步。

## 3. 为什么 `Psi_theta` 不接收条件

AM-PKNO 仍保留 PKNN / Stage3_0 的关键边界：

```text
Psi_theta(h_n)
```

而不是：

```text
Psi_theta(h_n, c_static)
```

原因是 parameterized Koopman 的核心证据来自：

```text
所有物理条件共享同一个 observable coordinate system，
不同条件只改变作用在该空间上的 Koopman operator。
```

如果把条件直接拼进 dictionary，模型会变成普通 conditional encoder，实验上难以判断提升来自 shared Koopman family，还是来自条件化表征本身。

## 4. 模块分解

### 4.1 Shared Dictionary

复用：

```text
src/pkno/dictionaries/shared_dictionary.py
```

形式：

```text
Psi_theta(h_n) = [1, handcrafted_basis(h_n), tanh(NN_theta(h_n))]
```

数据集默认：

| Dataset | `basis_kind` |
|---|---|
| Burgers | `burgers` |
| NS v1e-3 | `navier_stokes` |
| NS v1e-4 | `navier_stokes` |
| Shallow-water | `shallow_water` |

### 4.2 Condition Encoder

复用 Stage3_0 的条件拆分：

```text
c_static: 数据集/轨迹级显式条件
s_n:      从当前 history 提取的状态摘要
c_n:      ConditionEncoder(c_static, s_n)
```

`s_n` 包含：

```text
mean
std
rms energy
gradient rms
boundary mean/std
low/mid/high spectral energy ratios
```

### 4.3 AM Frequency Embedding

AM-FNO MLP 版用正交频率基缓解普通 MLP 的 spectral bias。Stage4_0 默认沿用 Chebyshev basis：

```text
e(k) = [k, T_1(k), T_2(k), ..., T_m(k)]
```

对应代码：

```text
src/ampkno/frequency.py
```

### 4.4 Conditioned Koopman Generator

1D：

```text
K(k, c_n) = G_phi(e(k), c_n)
```

输出 shape：

```text
[B, K, O, O]
```

2D full：

```text
K(kx, ky, c_n) = G_phi(e(kx, ky), c_n)
```

输出 shape：

```text
[B, Kx, Ky, O, O]
```

2D factorized 默认：

```text
K(kx, ky, c_n)[i,o] =
  sum_r Gx(kx, c_n)[i,o,r] * Gy(ky, c_n)[i,o,r]
```

因子 shape：

```text
Gx: [B, Kx, O, O, R]
Gy: [B, Ky, O, O, R]
```

数学上，factorized time marching 是：

```text
einsum("bixy,bxior,byior->boxy", z_hat, Gx, Gy)
```

但代码不能直接调用这个三操作数 einsum。PyTorch 的 contraction path 可能隐式 materialize：

```text
[B, Kx, Ky, O_in, O_out, R]
```

级别的中间张量。以 NS `B=10, X=64, Yh=33, O=32, R=1` 为例，单次 Koopman update 就可能额外产生约 `166 MiB` 的 complex 中间量。训练时还会展开：

```text
t_out * decompose = 40 * 8 = 320
```

次 Koopman update，autograd 会保存这些中间图，48GB GPU 会爆显存。

因此 `src/ampkno/operators.py` 中的 factorized path 使用 memory-efficient contraction：

```text
for r in rank:
  for i in observable_in:
    out += z_hat[:, i] * Gx[:, :, i, :, r] * Gy[:, :, i, :, r]
```

这样最大的临时张量保持在输出尺度：

```text
[B, O_out, Kx, Ky]
```

代价是速度会比直接 einsum 慢，但这是保留 all-frequency AM-PKNO 的更合理第一版实现。

另外，Stage4_0 默认对每次 Koopman update 开启 activation checkpoint：

```text
checkpoint_koopman = true
```

原因是 autoregressive training 会展开 `t_out * decompose` 次 Koopman update。checkpoint 会在 backward 时重算 operator forward，减少保存的中间激活。代价是训练更慢，但比单纯降低 batch size 或直接截断频率更符合 Stage4_0 要验证 all-frequency AM-PKNO 的目标。

如果后续确认某张 GPU 显存充足、希望换速度，可以在训练命令中加：

```text
--no-checkpoint-koopman
```

## 5. Shape 约定

1D：

```text
history:   [B, X, C_in]
condition: [B, C_cond]
z:         [B, X, O]
FFT z:     [B, O, K]
K:         [B, K, O, O]
pred:      [B, X, C_out]
```

2D：

```text
history:   [B, X, Y, C_in]
condition: [B, C_cond]
z:         [B, X, Y, O]
FFT z:     [B, O, Kx, Ky]
factor Gx: [B, Kx, O, O, R]
factor Gy: [B, Ky, O, O, R]
pred:      [B, X, Y, C_out]
```

当前四个数据集都是 scalar rollout，因此：

```text
C_out = 1
```

NS 和 shallow-water 的 `C_in` 等于 `t_in`。

## 6. 与 Stage1_0 / Stage3_0 的区别

| Model | Dictionary | Operator condition | Frequency policy | 2D cost control |
|---|---|---|---|---|
| Stage1_0 AM-KNO | pointwise encoder | default freq-only | all-frequency generator | factorized, freq-only |
| Stage3_0 Param-KNO | shared dictionary | `c_static + S(h_n)` | truncated modes + `K0 + DeltaK` | truncated modes |
| Stage4_0 AM-PKNO | shared dictionary | `c_static + S(h_n)` | AM all-frequency `K(k,c_n)` | conditioned factorized |

Stage4_0 的新增点是：

```text
2D factorized generator 不再是 freq-only，
而是 Gx(kx,c_n), Gy(ky,c_n) 都接收条件。
```

## 7. 默认超参数原则

### Burgers

```text
operator_size=32
decompose=8
max_modes=0
lr=1e-3
output_scale=0.02
```

### NS v1e-3

```text
t_in=10
t_out=40
ntrain=1000
ntest=200
operator_size=32
decompose=8
max_modes=0
operator_factorization=factorized
factorized_rank=1
lr=5e-4
output_scale=0.015
max_grad_norm=1.0
checkpoint_koopman=true
```

### NS v1e-4

```text
t_in=10
t_out=40
ntrain=1000
ntest=200
operator_size=32
decompose=8
max_modes=0
operator_factorization=factorized
factorized_rank=1
lr=3e-4
output_scale=0.01
max_grad_norm=1.0
checkpoint_koopman=true
```

注意：当前服务器记录显示 `ns_V1e-4_N10000_T30.mat` 实际有 50 帧，所以 Stage4_0 默认用 `t_out=40`。

### Shallow-water

```text
t_in=10
t_out=40
ntrain=900
ntest=100
dt=0.01
operator_size=32
decompose=4
max_modes=0
operator_factorization=factorized
factorized_rank=1
lr=5e-5
output_scale=0.005
max_grad_norm=0.1
checkpoint_koopman=true
```

这是根据 Stage3_0 shallow-water full run 的非有限 loss 风险做的保守默认。先保证 smoke/full 稳定，再逐步提高 `decompose` 或 `output_scale`。

## 8. 预期输出

每个 run 输出：

```text
outputs/stage4_0_am_pkno/<run_name>/
  args.json
  config.yaml
  env.txt
  metrics.csv
  rollout_error_by_step.csv
  spectral_metrics.csv
```

关键判断指标：

```text
test_full_rel_l2
test_step_rel_l2
high_band_rel_l2
gradient_rel_l2
rollout_error_by_step.csv 的增长趋势
```

AM-PKNO 成功的最低标准不是单步误差下降，而是：

```text
长 rollout 更稳
高频谱误差下降
gradient error 不变差
计算成本相对 full conditioned 2D generator 可接受
```

## 9. 当前限制

第一版 Stage4_0 仍有几个限制：

- 尚未加入显式 `K` 范数日志和 latent norm 日志；
- 尚未加入谱半径/谱范数稳定化约束；
- 尚未做 NS v1e-3 + v1e-4 joint training；
- shallow-water 的显式 dam 参数仍取决于 HDF5 是否提供元数据。
- memory-efficient factorized contraction 牺牲了一部分速度；如果 all-frequency 仍然太慢，优先做 `max_modes=16/24` 的 compute cap 对照，而不是回退到 full 2D generator。

这些应根据 smoke/full 结果决定是否进入 Stage4_1，而不是现在提前堆进第一版。
