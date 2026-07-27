# 模型设计决策记录

## D0：先 baseline，后模型设计

决定：在 KNO baseline 跑通之前，不实现完整 PKNO 架构。

原因：项目应从真实实验结果出发，尤其要先理解 rollout 稳定性和高频误差。

## D1：保持 Koopman 可解释性

决定：默认保留 observable / Fourier 空间中的线性 Koopman 演化。

原因：FFN、U-Net、非线性 residual 可能提升短期误差，但会削弱 Koopman 解释性。它们应通过消融实验进入。

## D2：官方代码优先

决定：KNO baseline 优先对齐 KoopmanLab。

原因：这样最小化与论文 baseline 的偏差。
