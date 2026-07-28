# Stage3_0 Shallow-water 稳定性记录

本文记录 Stage3_0 PKNO 在 shallow-water 数据集上的 smoke/full 现象，避免把后续非有限 loss 误判为数据格式问题。

## 1. Smoke 结果

使用最新模型和修正后的 shallow-water loader 后，smoke 已经稳定通过：

```text
epoch 0000 | 186.58s | train_full 1.937690e-01 | test_full 6.183325e-02
```

对应 CSV 核心指标：

```text
train_step_rel_l2 = 1.93144294e-01
train_full_rel_l2 = 1.93769035e-01
test_step_rel_l2  = 6.10953178e-02
test_full_rel_l2  = 6.18332484e-02
test_pred_mse     = 1.69727309e-01
test_recon_mse    = 2.54558547e-03
lr                = 2.00000000e-04
```

结论：数据读取、HDF5 layout 推断、condition vector、手工 shared dictionary、参数化 Fourier Koopman、autoregressive rollout 这条链路是可跑通的。旧 smoke 的 `nan` 更可能来自 loader/layout 或过强默认数值设置，而不是模型完全不可训练。

## 2. Full run: r8 默认保守版发散

命令关键超参：

```text
t_out = 40
decompose = 8
lr = 2e-4
delta_scale = 0.02
max_grad_norm = 0.5
```

日志片段：

```text
epoch 0000 | 163.33s | train_full 1.937690e-01 | test_full 6.183325e-02
epoch 0001 | 161.69s | train_full 6.124934e-02 | test_full 4.687385e-02
epoch 0002 | 161.86s | train_full 4.168359e-02 | test_full 3.572356e-02
epoch 0003 | 162.00s | train_full 4.093905e-02 | test_full 3.861276e-02
epoch 0004 | 159.85s | train_full 3.627376e-02 | test_full 3.397907e-02
epoch 0005 | 161.68s | train_full 3.446428e-02 | test_full 3.509692e-02
epoch 0006 | 160.44s | train_full 3.367735e-02 | test_full 4.280325e-02
epoch 0007 | 162.08s | train_full 3.413924e-02 | test_full 3.112087e-02
epoch 0008 | 161.70s | train_full 4.971689e-02 | test_full 7.800540e-02
epoch 0009 | 162.06s | train_full 8.093924e-02 | test_full 8.133300e-02
epoch 0010 | 161.28s | train_full 8.425776e-02 | test_full 8.639414e-02
epoch 0011 | 162.80s | train_full 8.124053e-02 | test_full 7.983323e-02
FloatingPointError: Non-finite training loss at epoch=12, batch=104.
```

判断：

- 前 7 个 epoch 有明显下降，最佳 `test_full_rel_l2` 约为 `3.112087e-02`。
- epoch 8 后误差抬升，随后在 epoch 12 出现非有限 loss。
- 这不是 `nohup` 或 shell 自己中断，而是训练保护逻辑主动停止。
- 如果该 run 启用了 `--save-checkpoint`，`checkpoint_best.pt` 仍有保留价值。

## 3. Full run: 更小 lr/delta 的 r8 仍发散

第二次尝试保持 `decompose=8`，但降低训练强度：

```text
t_out = 40
decompose = 8
lr = 1e-4
delta_scale = 0.01
max_grad_norm = 0.25
```

日志片段：

```text
epoch 0000 | 161.42s | train_full 3.043381e-01 | test_full 6.586779e-02
epoch 0001 | 160.77s | train_full 6.057446e-02 | test_full 6.813152e-02
epoch 0002 | 159.34s | train_full 5.355364e-02 | test_full 4.441727e-02
epoch 0003 | 160.68s | train_full 5.092808e-02 | test_full 7.045598e-02
FloatingPointError: Non-finite training loss at epoch=4, batch=18.
```

判断：

- 仅降低 optimizer 步长、`DeltaK` 幅度和梯度裁剪阈值，不足以解决 shallow-water full 的长期稳定性。
- 问题更像 forward rollout 中的 Koopman 递推深度不稳定，而不是单纯的 optimizer 更新过大。

## 4. 当前解释

Shallow-water 的默认 full 设置为：

```text
t_out = 40
decompose = 8
```

每个样本 rollout 近似包含：

```text
40 * 8 = 320
```

次 Fourier Koopman residual update。对 radial dam break 这种冲击前沿明显、边界影响强的 2D PDE，这个更新深度可能让 latent dynamics 在训练中逐步放大，最终产生 NaN/Inf。

因此当前结论应写为：

```text
shallow-water t40 + decompose=8 可以通过 smoke，早期 full 会下降，但 full 训练存在明显数值发散风险。
```

这更像 PKNO/KNO 稳定化问题，而不是数据格式问题。

## 5. 下一条推荐 full 命令

优先测试降低 Koopman 推进深度：

```bash
RUN=pkno_shallow_water_o32_m16_r4_t40_ep500_seed42_stable
CUDA_VISIBLE_DEVICES=3 nohup python -u experiments/stage3_0/train_pkno_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" \
  --run-name "$RUN" --output-dir outputs/stage3_0_param_kno \
  --epochs 500 --batch-size 5 --t-in 10 --t-out 40 \
  --ntrain 900 --ntest 100 \
  --operator-size 32 --modes 16 --decompose 4 \
  --dt 0.01 --lr 5e-5 --delta-scale 0.005 --max-grad-norm 0.1 \
  --seed 42 --save-checkpoint --device cuda \
  > logs/stage3_0_param_kno/$RUN.log 2>&1 &
```

若该设置仍发散，下一步应考虑显式稳定化设计，而不是继续只调学习率：

- 对 generated `DeltaK_k` 做谱范数或 Frobenius 范数约束。
- 在 Fourier Koopman update 后加入 spectral damping。
- 对 shallow-water 单独使用更小 `modes` 或 `decompose=2` 做稳定性边界扫描。
- 在训练指标中单独记录 `K_k` correction norm 和 latent norm，定位发散发生在 dictionary、Koopman update 还是 decoder。
