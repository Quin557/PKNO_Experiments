# IKNO Baseline Design

## Scope and sources

This baseline implements the central architecture in `ref/papers/2503.19717v1.pdf` in the current PyTorch rollout framework. It uses the repository's existing loaders for Burgers, Navier-Stokes v1e-3, Navier-Stokes v1e-4, and shallow-water, so dataset splits, time histories, targets, metrics, and output files remain comparable with the existing KNO-family runs.

No public IKNO implementation was available in the supplied references. `external/KoopmanLab` is used only to align the established KNO data and autoregressive conventions. The IKNO implementation itself is local and does not import or modify KoopmanLab.

## Implemented architecture

For channels-last history `h` of shape `[B, X, C_in]` or `[B, X, Y, C_in]`:

```text
z0 = G(h)                                      # pointwise invertible dictionary
zl+1 = GELU(IFFT(K^p * FFT(zl)) + Conv1x1(zl)) # l = 1 ... L
prediction = G_inverse(zL)[..., -1:]
```

`G` splits the input channels into two halves, zero-pads each half to `O / 2`, and applies residual additive coupling blocks. A coupling block is:

```text
[left, right] -> [right, left + H(right)]
```

where `H` is a residual three-linear-layer MLP with GELU activations. Its explicit inverse is:

```text
[right, updated_left] -> [updated_left - H(right), right]
```

Thus `G_inverse(G(h)) == h` up to floating-point arithmetic and without a separately trained decoder. The inverse can still be applied after a Koopman update; it removes the padding coordinates to return the physical time-delay channels, as specified by the INN construction in the paper.

`K` is a trainable complex matrix for each retained Fourier mode. The 2D implementation keeps the positive and negative x-frequency blocks and the non-negative rFFT y-frequency block, matching the KNO/IKNO Fourier layout. `p=2` applies each block matrix twice before the inverse FFT. The 1x1 convolution is the paper's high-frequency information path.

## Deliberate implementation choices

The paper fixes `O=32`, `M=16`, `L=4`, and `p=2`, but does not report an INN depth schedule `c_i` or the three-layer MLP hidden widths. The baseline therefore uses a constant valid schedule `c_i=16` for all four coupling blocks and `H` hidden width 128. These are explicit CLI/config parameters and must be reported for every run.

The paper's loss is rollout relative L2 and contains no reconstruction term. Existing repository trainers optimize weighted prediction MSE while reporting rollout relative L2, and all current baselines use this trainer. To preserve the comparison protocol, IKNO uses that same prediction objective with `recon_weight=0.0`; the returned reconstruction tensor is exactly the input only to satisfy the shared two-output interface. It contributes neither gradients nor loss.

The current datasets have a one-step Burgers target and 40-step NS/shallow-water targets. They therefore retain their existing loader and autoregressive settings rather than trying to reproduce the paper's separate 10-to-90 setup.

## Files

```text
src/ikno/invertible.py   # pointwise INN dictionary
src/ikno/operators.py    # fixed per-mode complex Fourier Koopman blocks
src/ikno/models.py       # 1D and 2D IKNO models
experiments/ikno/        # common trainer adapter and four dataset entry points
configs/model/ikno.yaml
configs/experiment/ikno/
```

## Comparison constraints

- Use `operator_size=32`, `modes=16`, `decompose=4`, and `koopman_power=2` for the paper-aligned primary baseline.
- Keep the current data split, `t_in`, `t_out`, batch size, seed, scheduler, output directory pattern, and metrics unchanged from the existing KNO baselines.
- Do not add a reconstruction weight. Any future INN-width or depth ablation must use a distinct run name.
