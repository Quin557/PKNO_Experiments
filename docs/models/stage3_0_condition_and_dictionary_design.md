# Stage3_0 条件与共享字典设计说明

## 1. 先澄清符号

Stage3_0 中不要把所有条件都叫成一个固定参数。更准确的拆法是：

```text
c_static: 数据集级或轨迹级已知条件
c_n:      第 n 个预测区间使用的条件
c_n_state: 从当前历史状态 h_n 提取的动态状态条件摘要
```

当前实现中的训练脚本传入模型的 `condition` 是显式有限维条件，主要对应 `c_static`。模型内部还会从当前 history 实时计算 `c_n_state`，两者合并后用于生成 Koopman 矩阵：

```text
z_n = Psi_theta(h_n)
c_n_state = S_theta(h_n)
K_k = K0_k + DeltaK_phi(freq(k), c_static, c_n_state)
z_{n+1,k} = K_k z_{n,k}
```

如果未来数据集中有真正随时间变化的 forcing、边界条件、控制量或外部输入，可以把它们作为每步不同的显式 `c_n` 传给模型。当前四个数据集多数没有完整暴露这些变量，因此先用：

```text
c_n ~= [c_static, c_n_state]
```

这是为了让 Stage3_0 先可跑、可比较，同时不把理论写死。

## 2. `c_static` 从哪里来

### Burgers

当前 KoopmanLab/FNO Burgers 文件通常只提供：

```text
a: 初始条件
u: 目标解
```

没有逐样本 viscosity、forcing 或边界条件字段。因此 Stage3_0 的显式条件只能来自实验设置和数据文件约定：

| 字段 | 来源 | 含义 |
|---|---|---|
| `log10_reynolds` | 文件名/数据卡中的 R10 约定 | 近似代表 Burgers 数据生成粘性/雷诺条件 |
| `dx` | `1 / (grid_size - 1)` | 网格 spacing |
| `sub` | 命令行 `--sub` | 下采样倍率 |
| `is_periodic` | 数据生成/loader 假设 | 周期边界标识 |

这些确实是弱条件。Burgers 在 Stage3_0 的主要作用不是证明强跨参数泛化，而是快速验证：

```text
1D complex K_k 生成是否稳定
shared dictionary + parameterized K 是否能训练
rollout/metrics/output 是否可用
```

真正的样本级差异主要来自 `c_n_state = S_theta(history)`。

### Navier-Stokes v1e-3 / v1e-4

NS 是当前最核心，因为 viscosity 是明确物理条件。

| 字段 | 来源 |
|---|---|
| `log10_viscosity` | 脚本名与数据文件：`1e-3` 或 `1e-4` |
| `dx`, `dy` | 读取后网格大小计算 |
| `dt` | 命令行，默认 `1.0`，表示离散采样间隔单位 |
| `t_in`, `t_out` | 命令行 |
| `sub` | 命令行 |

单独跑 v1e-3 或 v1e-4 时，viscosity 在一个 run 内是常量；这仍能验证代码稳定性，但不能充分证明参数插值。更强实验应是联合训练：

```text
train: v1e-3 + v1e-4
test: v1e-3/v1e-4 seen condition, and ideally unseen v=5e-4
```

### Shallow-water

当前 shallow-water HDF5 若没有 dam radius、basin geometry、边界参数等元数据，就只能使用可确认的条件：

| 字段 | 来源 |
|---|---|
| `dx`, `dy` | 网格大小计算 |
| `dt` | 命令行 |
| `t_in`, `t_out` | 命令行 |
| `sub` | 命令行 |
| `radial_dam_break_flag` | 当前文件 `2D_rdb_NA_NA.h5` 的任务类型标识 |

如果 HDF5 后续确认有更多元数据，应优先补：

```text
dam_radius
dam_center_x, dam_center_y
height_inside, height_outside
boundary_type
topography_or_bathymetry summary
```

## 3. `c_n_state` 怎么设计

`c_n_state` 由 `StateSummaryEncoder` 从当前 history 实时提取，不是固定参数。当前每个输入通道提取：

```text
mean
std
rms energy
gradient rms
boundary mean
boundary std
low-frequency energy ratio
mid-frequency energy ratio
high-frequency energy ratio
```

这对应你提到的：

```text
状态摘要含能量、梯度能量、频带能量
shallow_water 使用历史质量/能量、边界区域统计
```

实现位置：

```text
src/pkno/dictionaries/shared_dictionary.py
```

注意：这些摘要不进入 `Psi_theta`，只进入 `K_k` 生成器。这样能保留 shared dictionary 的定义。

## 4. 共享字典怎么设计

Stage3_0 的 shared dictionary 是：

```text
z_n = Psi_theta(h_n)
```

其中 `h_n` 是历史窗口。代码中使用点态共享映射：

```text
physical_lift(h_n) + learned_observables(h_n)
```

然后经过 `tanh` 得到 observable field。它是 KNO 风格的场字典，不是 PKNN 中 flat ODE state 的直接复制。

为什么这样设计：

1. PKNN 要求不同参数条件共享一个 observable space；
2. KNO 本来就在 encoder 后的 observable field 上做 Fourier-domain Koopman 推进；
3. PDE 数据在不同网格上仍应保留 neural operator 风格，所以字典采用 pointwise/shared mapping，而不是给每个网格点单独参数。

当前不把 `c` 输入字典：

```text
Psi_theta(h_n, c)   # 当前不采用
```

原因是这会让不同条件拥有不同坐标系，削弱 PKNN 的 shared dictionary 假设。Stage3_0 先验证最干净的版本：

```text
same Psi_theta, different K_k(c_n)
```

## 5. 理论依据

PKNN 论文处理 parametric dynamics：

```text
x_{n+1} = f(x_n, u_n)
Psi(x_{n+1}) ~= K(u_n) Psi(x_n)
```

关键不是 `u_n` 必须固定，而是每一步存在一个条件值或控制值，决定该步 Koopman operator。静态参数是特例：

```text
u_n = u, for all n
```

时变参数则是：

```text
u_n changes with n
```

Stage3_0 在 PDE/KNO 中对应：

```text
h_{n+1} = F_{c_n}(h_n)
Psi_theta(h_{n+1}) ~= K(c_n) Psi_theta(h_n)
```

KNO 的 Fourier 实现进一步把一个大矩阵拆成逐模态矩阵：

```text
z_hat_{n+1,k} = K_k(c_n) z_hat_{n,k}
```

因此当前设计的理论含义是：

```text
公共 observable field + 条件化逐频率 Koopman family
```

而不是“固定参数模型”。

## 6. 当前效果预期

最可能有效的顺序：

1. NS v1e-4 / v1e-3：因为 viscosity 是明确条件，最能体现 parameterized Koopman。
2. Shallow-water：如果 state summary 捕捉到质量、边界和能量变化，可能改善 rollout 稳定性。
3. Burgers：主要用于低成本验证与 1D sanity check，除非补充多 viscosity/R 数据，否则不是最强 PKNO 证据。

最值得补的后续实验：

```text
multi-condition NS:
  fixed KNO jointly on v1e-3 + v1e-4
  Param-KNO jointly on v1e-3 + v1e-4

multi-viscosity Burgers:
  generate or collect R/nu variants
  train on seen nu
  test on unseen/interpolated nu
```

这两个实验会比单文件训练更能突出 PKNO。
