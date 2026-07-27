# PKNN / Parametric Koopman PyTorch 迁移说明

本文件用于 Stage 3，不是当前 Stage 0 的执行入口。

## 原则

PKNN / Parametric Koopman 参考实现只作为算法参考，不直接混入 KNO baseline 训练循环。

## 未来目标模块

```text
src/pkno/dictionaries/shared_dictionary.py
src/pkno/operators/koopman_parameterized.py
src/pkno/models/param_kno.py
src/pkno/trainers/train_rollout.py
tests/test_pknn_port.py
```

## 需要映射的内容

| 参考概念 | PyTorch 目标 | 当前状态 |
|---|---|---|
| shared dictionary / observables | `SharedDictionary` | 未开始 |
| parameter-conditioned Koopman matrix | `ParameterizedKoopmanOperator` | 未开始 |
| rollout / composition evaluator | trainer/evaluator | 未开始 |
| prediction + reconstruction losses | losses module | 未开始 |

## 当前不做

- 不直接迁移 TensorFlow checkpoint；
- 不把参考 repo 混进训练依赖；
- 不在 KNO baseline 阶段实现 Param-KNO。
