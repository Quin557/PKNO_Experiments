# Stage3_0 条件与共享字典设计说明

## 1. 条件符号

Stage3_0 不把所有条件都理解成一个固定参数。当前实现采用下面的拆分：

```text
c_static:   数据集级或轨迹级已知条件，例如 viscosity、dx、dt、t_in
c_n:        第 n 次 Koopman 推进一步使用的条件
c_n_state:  从当前 history h_n 提取的动态状态摘要
```

训练脚本传入模型的 `condition` 对应有限维的 `c_static`。模型内部还从当前 history 实时计算 `c_n_state`，然后二者共同进入 Koopman 矩阵生成器：

```text
z_n = Psi_theta(h_n)
c_n_state = S_theta(h_n)
K_k(c_n) = K0_k + DeltaK_phi(freq(k), c_static, c_n_state)
z_hat_{n+1,k} = K_k(c_n) z_hat_{n,k}
```

所以当前可理解为：

```text
c_n = [c_static, c_n_state]
```

如果后续数据集提供显式时变 forcing、控制量或边界条件，可以把它们作为真正随步数变化的 `c_n` 传入，而不是只依赖 `c_n_state`。

## 2. 共享字典的边界

PKNN 参考实现的核心不是 TensorFlow 语法，而是结构：

```text
Psi(x) = [1, x, learned_dictionary(x)]
K = K(u)
```

因此 Stage3_0 的 PKNO 字典现在也显式包含常数项。代码位置：

```text
src/pkno/dictionaries/shared_dictionary.py
```

当前形式是：

```text
Psi_theta(h_n) = [1, handcrafted_basis(h_n), tanh(NN_theta(h_n))]
```

注意两点：

- `Psi_theta` 不接收 `c_static`，也不接收 `c_n_state`。
- 条件只进入 `K_k(c_n)` 的生成器。

这样做是为了保持同一套 observable 坐标系。若把 `c` 拼进字典，模型会变成普通 conditional encoder，实验上就很难区分提升来自共享 Koopman family，还是来自条件化表征本身。

## 3. 手工基函数

手工基函数是局部 field observable，输出仍是 channels-last 的 observable field。它们不是最终理论定稿，而是 Stage3_0 用于检验 PKNO 是否有效的实验性物理先验。

### Burgers

默认 `basis_kind=burgers`，输入通常是单通道初态或短历史。固定字典通道为：

```text
1
u
u^2
u^3
sin(pi u)
cos(pi u)
u_x
u_xx
u * u_x
```

理由：

- `1, u, u^2, u^3` 是 Koopman/EDMD 常见低阶 observable。
- `sin(pi u), cos(pi u)` 给 1D 周期或近周期结构一个 bounded 非线性基。
- `u_x, u_xx, u u_x` 对应 Burgers 中平流和扩散的局部结构。
- Burgers 当前文件里的 `R10/sub/dx` 更像弱条件，真正样本差异主要来自 `h_n`。

### Navier-Stokes v1e-3 / v1e-4

默认 `basis_kind=navier_stokes`。当前 NS 数据以标量涡量场历史作为输入，固定字典通道为：

```text
1
w_n
mean_t(w)
std_t(w)
w_n^2
w_n - w_0
w_x
w_y
sqrt(w_x^2 + w_y^2)
Delta w
```

理由：

- NS 是 Stage3_0 核心，因为 `log10(viscosity)` 是明确物理条件。
- `w_x, w_y, Delta w` 让 shared dictionary 直接看到局部梯度和粘性项相关结构。
- `w_n^2` 和梯度模长帮助表达能量、涡量强度和高频增长。
- `mean_t/std_t/w_n-w_0` 让同一个 `Psi_theta` 在 rollout 中感知历史窗口变化，而不是只看最后一帧。

### Shallow-water

默认 `basis_kind=shallow_water`。当前使用水深标量场，固定字典通道为：

```text
1
h_n
mean_t(h)
std_t(h)
h_n^2
sqrt(h_x^2 + h_y^2)
Delta h
boundary_mask
boundary_mask * h_n
h_n - mean_xy(h_n)
```

理由：

- PDEBench radial dam break 的强信号来自初始水深分布、冲击前沿和边界区域。
- `h_n^2` 可作为浅水能量代理，`h_n - mean_xy(h_n)` 对应局部质量偏差。
- `boundary_mask` 和 `boundary_mask*h_n` 让字典显式暴露边界区域，而不是全靠 MLP 学出来。
- 如果 HDF5 后续确认含 `dam_radius/dam_center/height_inside/height_outside/boundary_type`，这些应优先进入 `c_static`。

## 4. `c_static` 来源

### Burgers

```text
log10_reynolds: 从 burgers_data_R10.mat 文件约定得到
dx:             1 / (grid_size - 1)
sub:            命令行下采样率
is_periodic:    当前 loader 假设为 1
```

这些是弱条件。Burgers 在 Stage3_0 更适合做 1D sanity check，而不是最强的跨参数泛化证据。

### Navier-Stokes

```text
log10_viscosity: v1e-3 或 v1e-4
dx, dy:          读取后的网格 spacing
dt:              命令行采样间隔
t_in, t_out:     history 和 rollout 长度
sub:             下采样率
```

单独训练 v1e-3 或 v1e-4 时，viscosity 在一个 run 内是常量。更强的 PKNO 实验应做联合训练：

```text
train: v1e-3 + v1e-4
test:  seen viscosity, ideally unseen v=5e-4
```

### Shallow-water

```text
dx, dy:                  网格 spacing
dt:                      默认 0.01, 来自 PDEBench radial dam break 配置
t_in, t_out:             history 和 rollout 长度
sub:                     下采样率
radial_dam_break_flag:   当前任务类型标识
```

## 5. `c_n_state` 摘要

`StateSummaryEncoder` 从当前 history 中提取每个输入通道的统计量：

```text
mean
std
rms energy
gradient rms
boundary mean
boundary std
low/mid/high spectral energy ratios
```

这些摘要不属于 `Psi_theta`，而是进入 `K_k(c_n)`。这让同一个共享字典可以保留固定坐标，而 Koopman operator 随当前状态和物理条件变化。

## 6. Shallow-water smoke NaN 的判断

四个 smoke 中只有 shallow-water 出现：

```text
epoch 0000 | train_full nan | test_full nan
```

已处理的高风险点：

- loader 现在显式支持 grouped PDEBench 格式 `0000/data: (T,X,Y,C)`。
- loader 也支持转换后的 root `/data` 格式：`(B,X,Y,T)`, `(B,T,X,Y)`, `(B,X,Y,T,C)`, `(B,T,X,Y,C)`。
- 旧版对 4D/5D root 数据只靠维度大小判断，可能把 `(B,X,Y,T,C)` 误读成 `(B,T,X,Y,C)`。
- loader 会在训练前检查 shallow-water tensor 是否含 NaN/Inf，并给出明确错误。
- 训练循环现在遇到非有限 loss 会直接报错，不再静默写入 `nan` 指标。
- shallow-water 默认改为更保守的 `lr=2e-4`, `delta_scale=0.02`, `max_grad_norm=0.5`, `dt=0.01`。

如果服务器上仍然报非有限数据，优先检查 HDF5：

```bash
python - <<'PY'
import h5py, numpy as np, os
path = os.path.join(os.environ["DATA_ROOT"], os.environ["SHALLOW_WATER_FILE"])
with h5py.File(path, "r") as f:
    print("keys:", list(f.keys())[:5])
    if "data" in f and isinstance(f["data"], h5py.Dataset):
        d = f["data"]
        a = d[: min(2, d.shape[0])]
        print("/data shape:", d.shape, "finite:", np.isfinite(a).all())
    else:
        k = sorted(x for x in f.keys() if isinstance(f[x], h5py.Group) and "data" in f[x])[0]
        a = f[f"{k}/data"][:]
        print(k + "/data shape:", a.shape, "finite:", np.isfinite(a).all())
PY
```
