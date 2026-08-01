# Stage3_1 PKNO-U 模型设计

Stage3_1 的代号为 `stage3_1_param_kno_u`，模型名为 PKNO-U。它是独立于
Stage3_0 Param-KNO 和 Stage4_0 AM-PKNO 的新路线，目标是同时研究：

```text
1. U-Net 是否能补偿参数化 Koopman 频域更新遗漏的局部高频；
2. 物理条件与状态条件应如何分工，才能保持长 rollout 稳定。
```

## 1. 共享坐标与参数化动力学

三个模型都使用同一个、条件无关的字典：

```text
z_n = Psi_theta(h_n)
```

`c_static` 不会拼入 `Psi_theta`。它只决定频率矩阵：

```text
K_k = K_k(c_phys)
```

这保留了 PKNN 的共同 observable coordinate 假设。当前四个 loader 中已有的
网格、步长、粘度和任务元数据继续作为 `c_phys`。只有在同一训练数据中这些量确实
变化时，才能将结果解释为跨物理参数泛化；单独的 `nu=1e-3` 或 `nu=1e-4` run 不能
提供该结论。

## 2. 三个条件模型

| 名称 | `K_k` 条件 | 地位 |
|---|---|---|
| A `physical_only` | `c_phys` | 主模型 |
| B `physical_compact_state` | `c_phys` + 紧凑状态慢变量 | 消融 |
| C `physical_gated_state` | `c_phys` + 有界门控状态修正 | 消融 |

主模型 A 不从 history 为矩阵生成器提取动态摘要。B 将每个 history 通道的均值、标准差、
RMS、梯度 RMS 及低/高频能量比例经 LayerNorm 和小 MLP 压缩后再使用。C 进一步令：

```text
c_n = E_phys(c_phys) + sigmoid(g(s_n)) * E_state(s_n)
```

因此状态只能受限地修正物理条件主导的矩阵，不能像 Stage3_0 一样每一步无约束地
重新塑造全部 `K_k`。

## 3. 稳定参数化

Stage3_1 仍生成复数频率矩阵：

```text
D_k = K0,k + DeltaK_phi(k, c_n)
K_k = rho_max * (I + D_k) / (1 + ||D_k||_F)
```

默认 `rho_max=0.98`。因为 `||I + D_k||_2 <= 1 + ||D_k||_F`，每个生成的
transition 都满足：

```text
||K_k||_2 <= ||K_k||_F <= rho_max
```

每个内部 Koopman layer 直接应用 `z <- K_k z`，而不是旧式 `z <- z + K_k z`；否则约束
残差矩阵本身并不能约束真实的 `I + K_k` transition。这不是物理正确性的保证，但可以阻止
生成矩阵本身在重复相乘时成为无界放大源。对于
确实需要短时增长的任务，可通过 `--max-operator-norm` 显式放宽上界，不能默默取消。

## 4. U-Fourier 高频分支

原始 U-FNO 在 Fourier 更新中并联 U-Net 局部支路。PKNO-U 保持这一思想，但为避免
在 `t_out * decompose` 展开中重复堆叠过多 U-Net，采用前半段纯 Koopman、后半段
U-Fourier 的方式：

```text
z <- K(z, c_n)                                      # 前半段
z <- K(z, c_n) + beta * HighPass(U-Net(z))          # 后半段
```

`HighPass` 会删除 U-Net 输出中的低频部分，只让该支路补偿高频。最终的 1x1 skip
仍被保留，它是 U-FNO 公式中的通道线性项，不应被误当作高频卷积支路删除。

默认：

```text
unet_start_layer = decompose // 2
hf_cutoff = 0.5
hf_residual_scale = 0.05
```

Shallow-water 从 `decompose=2` 和 `hf_residual_scale=0.02` 开始。U-Net 使用
activation checkpoint，优先控制显存而非追求最高吞吐量。

## 5. 时间复合边界

主训练保持仓库既有 autoregressive MSE 流程：解码结果回填 history 后再预测下一步。
这保证与旧实验可比，但它不是“只编码一次”的纯 latent product。模型同时提供
`latent_ordered_rollout` 接口给 A：history 只编码一次，固定物理条件矩阵按顺序重复推进。
B/C 由于显式从预测状态重新计算条件，接口会拒绝执行，且不应被表述为固定线性 Koopman
演化。

## 6. 训练与诊断

训练损失保持与历史 PKNO 一致：

```text
L = 5 * rollout prediction MSE + 0.5 * reconstruction MSE
```

RL2 不参与默认优化，而是作为主评估指标。每个运行除原有文件外还会生成：

```text
stability_diagnostics.csv
```

该文件按 rollout step 记录 condition gate、矩阵 spectral norm、latent RMS 和
U-Net 高通残差 RMS，用于定位长时发散来源。
