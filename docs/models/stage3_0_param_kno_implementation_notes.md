# Stage3_0 Param-KNO 实现说明

## 1. PyTorch-only 约束

PKNN 参考仓库是 TensorFlow/Keras 实现，KNO 和本项目训练体系是 PyTorch。Stage3_0 严格使用 PyTorch：

```text
不 import tensorflow
不 import keras
不迁移 checkpoint
不混用 TF tensor 与 torch tensor
```

PKNN 只提供算法结构：

```text
shared dictionary
parameter-conditioned Koopman matrix K(u)
latent Koopman consistency loss
```

## 2. TF 到 PyTorch 的对应关系

| PKNN TensorFlow concept | Stage3_0 PyTorch implementation |
|---|---|
| `tf.keras.layers.Layer.call()` | `torch.nn.Module.forward()` |
| `Dense` | `torch.nn.Linear` |
| `tf.concat` | `torch.cat` |
| `tf.reshape` | `torch.reshape` / `.reshape()` |
| `tf.einsum("ij,ijk->ik")` | `torch.einsum(...)` with explicit batch/frequency axes |
| `K_u = NN(u)` | `DeltaK_phi(freq(k), c_static, c_n_state)` |
| `PsiNN(x)` | `SharedPointwiseDictionary(history)` |

The original PKNN code sets TensorFlow float type to `float64`. Stage3_0 uses PyTorch `float32` for real tensors and `torch.cfloat` for Fourier Koopman matrices, matching the KNO implementation style.

Condition terminology is documented in:

```text
docs/models/stage3_0_condition_and_dictionary_design.md
```

## 3. Shape conventions

All physical fields use channels-last layout.

1D:

```text
history:   [B, X, C_in]
target:    [B, X, T_out]       # scalar rollout
condition: [B, C_cond]
latent:    [B, X, O]
FFT:       [B, O, K]
K_k:       [B, modes, O, O]
```

2D:

```text
history:   [B, X, Y, C_in]
target:    [B, X, Y, T_out]    # scalar rollout
condition: [B, C_cond]
latent:    [B, X, Y, O]
FFT:       [B, O, Kx, Ky]
K_k:       [B, modes_x, modes_y, O, O]
```

## 4. Why `Psi_theta` does not receive `c`

PKNN's key idea is a parameter-independent observable space. Stage3_0 therefore keeps:

```text
z = Psi_theta(history)
```

not:

```text
z = Psi_theta(history, c)
```

The condition enters only here:

```text
K_k = K0_k + DeltaK_phi(freq(k), c_static, c_n_state)
```

This distinction matters experimentally. If the dictionary changes with `c`, it becomes unclear whether improvements come from a shared parameterized Koopman family or from an ordinary conditional representation.

## 5. Current limitations

- Stage3_0 does not yet implement a strict PKNN constant observable row such as `(1, 0, ..., 0)`.
- Stage3_0 uses direct complex matrix generation, not low-rank or basis-expanded `K(u)`.
- Shallow-water currently uses available grid/task metadata plus state summary; richer dam/boundary parameters should be added if present in the HDF5 metadata.
- Single-file NS runs have constant viscosity within each run. A joint v1e-3/v1e-4 experiment is recommended next for a stronger parameterized Koopman test.

## 6. Files

```text
src/pkno/dictionaries/shared_dictionary.py
src/pkno/operators/koopman_parameterized.py
src/pkno/models/param_kno.py
src/pkno/trainers/train_rollout.py
src/pkno/data/stage3_loaders.py
```
