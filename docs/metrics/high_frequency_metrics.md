# 高频指标说明

高频指标从 Stage 0 就开始准备，因为后续 KNO 优化和 PKNO 设计很可能不只看单步误差，而要看模型是否改善小尺度结构、频谱误差和长时间 rollout 稳定性。

## 基础指标

每次 baseline 运行至少记录：

- `MSE`
- `step relative L2`
- `full rollout relative L2`
- `rollout error by step`
- 参数量
- 每 epoch 时间
- 总训练时间

## 频谱指标

对预测 `pred` 和真实值 `target` 在空间维度做 FFT，然后按频率半径分成低频、中频、高频。

推荐频带：

```text
low:  0%  - 33%
mid:  33% - 66%
high: 66% - 100%
```

推荐输出：

- `low_band_spectral_rel_l2`
- `mid_band_spectral_rel_l2`
- `high_band_spectral_rel_l2`
- `high_frequency_energy_ratio_error`
- `gradient_rel_l2`

## 如何解释

- 高频误差下降是好事，但必须同时检查 rollout 是否更稳定。
- 只提升单步误差、不提升长 rollout，不一定说明模型更好。
- 高频能量不能明显变得不物理，否则可能只是过拟合噪声。
- 如果数据是多变量，应尽量按变量分别报告频谱误差。

## 分析模板

每个完整 run 建议写一段简短分析：

```text
Run:
Dataset:
Best full Rel L2:
High-band spectral error:
Gradient Rel L2:
Rollout stability:
Main observation:
Next action:
```
