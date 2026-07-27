# Equation Note: PKNO Forward

Final candidate direction:

```text
z_t = Psi_theta(u_t, x, c)
u_embed = E_theta(u_t)
K_k = G_phi(e(k), u_embed, c, bc)
z_{t+1,k} = K_k z_{t,k}
u_pred = D_theta(z_{t+1})
u_final = u_pred + optional_high_frequency_residual
```

Do not implement this until Stage 0 baselines and early ablations have evidence.
