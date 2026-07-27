# Equation Note: KNO Forward

KNO treats PDE solution evolution as nonlinear dynamics in physical space and approximately linear evolution in learned observable space.

```text
z_t = Psi_theta(u_t)
z_t_hat = FFT(z_t)
z_{t+1,k}_hat = K_k z_{t,k}_hat
z_{t+1} = IFFT(z_{t+1}_hat)
u_{t+1} = D_theta(z_{t+1} + high_frequency_complement)
```

The baseline should preserve this structure before adding nonlinear adapters.
