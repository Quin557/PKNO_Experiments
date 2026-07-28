# Stage3_0 PKNO 实验设计

Stage3_0 的代号是：

```text
stage3_0_param_kno
```

目标是绕过 Stage 1/2，先验证 PKNN 的 parameterized Koopman 思想能否以 PyTorch 方式融入 KNO：

```text
shared dictionary Psi_theta(h_n)
parameterized Fourier Koopman K_k(c_n)
```

## 1. 核心问题

Stage 0 已经证明四个 KNO baseline 能跑通：

- `nsv1e3`
- `nsv1e4`
- `burgers`
- `shallow_water`

Stage3_0 不重复证明 KNO 可训练，而是回答：

```text
在同一个 observable space 里，让每个 Fourier mode 的 Koopman matrix
随物理条件和当前状态摘要变化，是否比 fixed KNO 更稳。
```

## 2. 从 PKNN 到 KNO

PKNN 参考实现是 TensorFlow/Keras，但 Stage3_0 是 PyTorch-only。只迁移算法结构：

```text
Psi(x) = [1, x, learned_dictionary(x)]
K = K(u)
Psi(y) ~= K(u) Psi(x)
```

KNO 的更新发生在 Fourier 域：

```text
z_n = Psi_theta(h_n)
z_hat_n = FFT(z_n)
z_hat_{n+1,k} = K_k z_hat_{n,k}
u_{n+1} = D_theta(IFFT(z_hat_{n+1}))
```

Stage3_0 合并后使用：

```text
z_n = Psi_theta(h_n)
c_n_state = S_theta(h_n)
K_k(c_n) = K0_k + DeltaK_phi(freq(k), c_static, c_n_state)
z_hat_{n+1,k} = K_k(c_n) z_hat_{n,k}
u_pred = D_theta(IFFT(z_hat_{n+1}))
```

关键边界：

```text
Psi_theta 不接收 c_static 或 c_n_state
K_k 接收 c_static 和 c_n_state
```

这样才能保持 PKNN 的 shared dictionary / common observable space 假设。

## 3. 当前共享字典

代码位置：

```text
src/pkno/dictionaries/shared_dictionary.py
```

当前字典是显式固定基函数加学习基函数：

```text
Psi_theta(h_n) = [1, handcrafted_basis(h_n), tanh(NN_theta(h_n))]
```

这回答了“有没有常数项”的问题：有。第一个固定 observable channel 是常数 `1`。这比旧版 `physical_lift + learned_observables` 更接近 PKNN 原实现，也更容易解释。

各数据集默认基函数：

| Dataset | `basis_kind` | Fixed basis |
|---|---|---|
| Burgers | `burgers` | `1, u, u^2, u^3, sin(pi u), cos(pi u), u_x, u_xx, u u_x` |
| NS v1e-3/v1e-4 | `navier_stokes` | `1, w_n, mean_t(w), std_t(w), w_n^2, w_n-w_0, w_x, w_y, |grad w|, Delta w` |
| Shallow-water | `shallow_water` | `1, h_n, mean_t(h), std_t(h), h_n^2, |grad h|, Delta h, boundary_mask, boundary_mask*h_n, h_n-mean_xy(h_n)` |

`operator_size=32` 时，剩余通道由 learned dictionary 填充。若把 `operator_size` 设得太小，小于固定基函数通道数，代码会直接报错。

## 4. 条件设计

更完整说明见：

```text
docs/models/stage3_0_condition_and_dictionary_design.md
```

简表：

| Dataset | Explicit condition `c_static` | Dynamic condition `c_n_state` | 目的 |
|---|---|---|---|
| Burgers | `log10_reynolds, dx, sub, is_periodic` | history 的均值、能量、梯度、边界、频带摘要 | 1D complex K_k sanity check |
| NS v1e-3 | `log10(viscosity), dx, dy, dt, t_in, t_out, sub` | 能量、梯度能量、频带能量等状态摘要 | 核心实验，验证粘性条件 |
| NS v1e-4 | 同上，`viscosity=1e-4` | 同上 | 更难的低粘性条件 |
| Shallow-water | `dx, dy, dt, t_in, t_out, sub, radial_dam_break_flag` | 水深质量/能量代理、边界统计、频带摘要 | 检验初态和边界主导的 2D PDE |

## 5. Shallow-water smoke 结果处理

当前 smoke 结果：

```text
Burgers:       train_full 4.800567e-01 | test_full 2.680123e-01
NS v1e-3:      train_full 5.304509e-01 | test_full 4.350000e-01
NS v1e-4:      train_full 6.207223e-01 | test_full 4.801318e-01
Shallow-water: train_full nan          | test_full nan
```

对 shallow-water 已做三类修正：

- HDF5 layout 推断更严格，覆盖 grouped PDEBench 原格式和 root `/data` 转换格式。
- 数据进入训练前检查 NaN/Inf。
- shallow-water 默认超参更保守：`lr=2e-4`, `delta_scale=0.02`, `max_grad_norm=0.5`, `dt=0.01`。

如果新 smoke 直接报 `non-finite values`，说明优先是数据或转换文件问题。如果数据 finite 但训练 loss 非有限，再优先减小：

```text
--lr 1e-4
--delta-scale 0.01
--decompose 4
```

## 6. 推荐对照

第一轮结论不要只看是否超过所有 baseline，更应该看趋势：

| 对照 | 判断 |
|---|---|
| Stage0 KNO vs Stage3_0 PKNO | 参数化 `K_k` 是否降低 full rollout 或高频误差 |
| NS v1e-3 vs NS v1e-4 | viscosity 条件是否被模型稳定吸收 |
| Fixed dictionary ablation | 手工基函数是否改善早期稳定性 |
| Shallow-water smoke/full | layout 修复后是否还出现 NaN 或边界误差 |

更典型的后续 PKNO 实验：

```text
train on nu = {1e-3, 1e-4}
test on unseen nu = 5e-4
```

如果暂时没有中间 viscosity 数据，也可以先做联合训练：

```text
fixed KNO jointly trained on v1e-3 + v1e-4
Param-KNO jointly trained on v1e-3 + v1e-4 with log10(nu)
```
