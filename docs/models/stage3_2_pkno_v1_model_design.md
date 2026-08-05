# Stage3_2 PKNO_v1 Model Design

## Scope

`stage3_2_pkno_v1` is an independent route. It does not overwrite
`stage3_0_param_kno`, does not use PKNO-U, and does not include a U-Net. The
Fourier Koopman update, shared dictionary, 1x1 convolutional skip, observable
size `O=32`, and `modes=16` remain aligned with the existing PKNO comparison.

## Model

The shared dictionary remains condition independent:

```text
z_n = Psi_theta(h_n)
```

The current PKNO concatenates physical metadata and a state summary before one
condition MLP. PKNO_v1 instead forms a physics-first embedding:

```text
e_phys  = E_phys(normalize(c_physics))
e_state = tanh(E_state(S(h_n)))
g_n     = 0.1 * sigmoid(G(e_phys, e_state))
e_n     = e_phys + g_n * e_state
K_k     = K0_k + DeltaK_phi(freq(k), e_n)
```

The state path stays at 64 dimensions, but its influence is bounded by a
learned gate. Only one Fourier weight tensor is generated per model step;
PKNO_v1 does not generate separate full physics/state matrices.

The physical condition vector excludes `t_in` and `t_out`, because they are
training protocol values rather than properties of the one-step PDE dynamics.

| Dataset | PKNO_v1 physical condition |
|---|---|
| Burgers | `log10_reynolds, dx, sub, is_periodic` |
| Navier-Stokes | `log10_viscosity, dx, dy, dt, sub` |
| Shallow-water | `dx, dy, dt, sub, radial_dam_break_flag` |

## Residual Field Prediction

PKNO_v1 predicts the physical increment:

```text
u_(n+1) = u_n + D_theta(z_(n+1))
```

The existing pointwise decoder and 1x1 convolutional skip are retained. This
is an output residual, not a U-Net or a multiscale convolution branch. The
`--direct-prediction` flag is the ablation that removes this residual.

## Soft Stability

Training retains MSE as its primary objective. Relative L2 is a validation and
reporting metric.

```text
L = 5 L_prediction_MSE + 0.5 L_reconstruction_MSE
  + lambda_state ||g_n * e_state||^2
  + lambda_smooth ||g_n e_state(n) - stopgrad(g_(n-1)e_state(n-1))||^2
  + I[nu=1e-4] lambda_growth max(0, RMS(u_pred)/RMS(u_n) - q_0.99)^2
```

`q_0.99` is measured only from the training trajectories. The growth term is
soft and does not impose monotone decay, since forced Navier-Stokes dynamics
need not have monotone energy. Full matrix SVD and full adjacent-matrix losses
are deliberately excluded: at `T_out=40`, `decompose=8`, they add avoidable
memory pressure without providing a cleaner physical signal.

## Curriculum

For NS and shallow-water, all runs use 500 epochs in total:

```text
0-49:    one-step prediction from random teacher-forced history windows
50-74:   5-step autoregressive rollout
75-99:   10-step autoregressive rollout
100-499: full T40 autoregressive rollout
```

Burgers is already a one-step task and uses 500 one-step epochs.

## Expected Effect and Risk

The changes target the observed failure modes rather than guarantee a result:

- Shallow-water: residual field prediction and teacher-forced warm-up target
  its large first-step error.
- NS `nu=1e-3`: residual prediction plus rollout curriculum target both local
  and accumulated error.
- NS `nu=1e-4`: bounded state modulation and the growth envelope target rapid
  long-rollout error growth.
- Burgers: the Stage3_0 PKNO is already strongest, so `--direct-prediction`
  remains a required ablation to detect a residual-regression case.

No PKNO_v1 result may replace the paper PKNO result unless all four historical
full-rollout RL2 values improve at seed 42.
