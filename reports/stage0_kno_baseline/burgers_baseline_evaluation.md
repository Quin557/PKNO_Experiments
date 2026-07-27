# Burgers Baseline 评估

## Run

```text
kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42
```

## 来源

服务器上传的日志与结构化输出：

```text
logs/stage0_kno_baseline/kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42.log
outputs/stage0_kno_baseline/kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42/
```

## 配置

```text
epochs: 500
batch_size: 64
operator_size: 32
modes: 16
decompose: 8
sub: 32
lr: 0.001
seed: 42
params: 33,921
```

## 结果

结构化结果：

```text
test_mse = 2.8737262709910283e-05
```

从日志解析得到：

```text
epochs: 500
first_eval_pred_mse: 1.8463654210791e-02
best_epoch: 499
best_eval_pred_mse: 2.87372627099103e-05
last_epoch: 499
last_eval_pred_mse: 2.87372627099103e-05
last_eval_reconstruction_mse: 3.28313068166608e-04
average_epoch_time: 0.1792 s
```

## 判断

这个 Burgers baseline 是可用结果。

理由：

- 训练完整跑完 500 epoch；
- eval prediction MSE 从 `1.846e-2` 稳定下降到 `2.874e-5`；
- 最佳 epoch 是最后一轮，说明没有看到明显后期退化；
- 参数量较小，训练速度很快，适合作为后续高频指标和小结构消融的快速验证数据。

## 后续建议

下一步不要重复跑同一配置。建议补：

```text
spectral_metrics.csv
rollout / mesh-independence 相关评估
```

然后再把 Burgers 用作 Stage 1/2 的快速调试数据。
