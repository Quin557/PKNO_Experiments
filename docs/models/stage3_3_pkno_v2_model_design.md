# Stage3_3 PKNO_v2-A Model Design / Stage3_3 PKNO_v2-A 模型设计

## Scope / 范围

PKNO_v2-A is an independent route under `src/pkno_v2`; it does not modify
PKNO, PKNO-U, or PKNO_v1.  The comparison contract keeps `O=32`, `modes=16`,
seed 42, NS split 1000/200, and shallow-water split 900/100.

PKNO_v2-A 是 `src/pkno_v2` 下的独立路线，不修改旧 PKNO、PKNO-U 或
PKNO_v1。对比协议固定 `O=32`、`modes=16`、seed 42，NS 使用 1000/200，
shallow-water 使用 900/100。

## Parameterized operator / 参数化算子

The operator uses a low-rank physical family:

```text
K_k(c_n) = K0_k + U_k diag(alpha_k(c_n)) V_k^H,   rank = 4
```

`c_n` is the normalized explicit physical vector plus a compact 16-D state
summary.  The state path is gated by at most 0.15, so physical metadata remains
the anchor.  `alpha` is generated for every retained frequency, while `U,V`
are shared frequency factors.  This exposes the parameter family without
materializing a full batch-specific O×O generator tensor.

`c_n` 由归一化的显式物理条件和紧凑的 16 维状态摘要组成；状态支路的门控
上限为 0.15，物理条件始终是锚点。每个保留频率只生成 rank=4 的系数，
`U,V` 为共享频率因子，避免生成完整的 batch-specific O×O 张量。

Each decomposition step is a bounded Euler update:

```text
z_(l+1) = z_l + eta_max * sigmoid(raw_eta) / decompose * K_k(c_n) z_l
```

This is a soft bound rather than a hard spectral projection.  It keeps the
step size explicit and reduces long-rollout amplification while preserving
learnable dynamics.

每个 decompose step 使用有界 Euler 更新。它不是硬谱投影，而是显式限制步长，
在保留可学习动力学的同时降低长时间推进放大。

## High-frequency path / 高频路径

The Fourier branch remains the main predictor.  A single lightweight 3×3
convolutional residual is applied to the latest physical frame and initialized
at scale 0.01.  There is no U-Net; the branch is intentionally small to avoid
the memory multiplication seen when a multiscale network is repeated inside
`t_out × decompose`.

Fourier 分支仍是主预测器；对最新物理场增加一个轻量 3×3 卷积残差，初始
幅度约 0.01。不使用 U-Net，避免在 `t_out × decompose` 展开中重复多尺度网络
造成显存和稳定性问题。

## Objective / 训练目标

MSE remains the optimization target for continuity with all existing baselines,
but prediction MSE is linearly weighted from 1.0 at the first step to 2.0 at the
last step.  Reconstruction MSE has weight 0.5.  Small gradient and high-band
losses (1e-3 and 1e-4) and a gate/step penalty provide auxiliary signals.  RL2
is validation/reporting only, never the training target.

为保持与现有基线一致，训练目标仍是 MSE；预测 MSE 按 rollout step 从 1.0 线性
增加到 2.0，重建 MSE 权重为 0.5。梯度、高频和门控/步长项权重分别为 1e-3、
1e-4、1e-4，仅作为辅助项。RL2 只用于验证、正式报告和 promotion 判断，不作为
训练目标。

## Curriculum and promotion / 课程与晋级

For T=40, epochs 0–39 use one teacher-forced step, 40–79 use five steps,
80–119 use ten steps, and 120–499 use full autoregressive rollout.  Burgers
uses one step throughout.  Smoke is exactly one epoch; it must finish without
non-finite loss before the 500-epoch run.

Promotion requires all four full-rollout RL2 values to beat old PKNO.  The
stretch target is to also beat KNO, iKNO, and AM-KNO under matched protocols.

对于 T=40，0–39 epoch 为 teacher-forced 单步，40–79 为 5 步，80–119 为 10 步，
120–499 为完整自回归 rollout；Burgers 始终单步。Smoke 固定 1 epoch，必须无
NaN/Inf 后才能跑 500 epoch。四项 RL2 全部优于旧 PKNO 才能替换论文结果；进一步
的 stretch target 是在协议匹配时同时超过 KNO、iKNO、AM-KNO。
