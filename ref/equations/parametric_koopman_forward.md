# Equation Note: Parametric Koopman Forward

Stage 3 planned idea:

```text
z_t = Psi_theta(u_t, c)
K(c) = G_phi(c)
z_{t+1} = K(c) z_t
u_{t+1} = D_theta(z_{t+1})
```

For Fourier-domain PKNO:

```text
K_k(c) = G_phi(e(k), c)
z_{t+1,k} = K_k(c) z_{t,k}
```
