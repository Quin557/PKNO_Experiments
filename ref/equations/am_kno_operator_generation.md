# Equation Note: Frequency-Conditioned Koopman Generation

Stage 1 planned idea:

```text
K_k = G_phi(e(k))
z_{t+1,k} = K_k z_{t,k}
```

State-conditioned variant:

```text
K_k = G_phi(e(k), E_theta(u_t))
z_{t+1,k} = K_k z_{t,k}
```

This is not part of Stage 0 baseline reproduction.
