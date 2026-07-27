# Code Note: AM-Style Frequency Generation

## Role

This note is for later Stage 1 model design only. It is not part of the current KNO baseline reproduction entry.

## Idea

Instead of learning an independent Koopman matrix for each retained frequency, learn a shared generator:

```text
K_k = G_phi(e(k))
K_k = G_phi(e(k), u_embed)
```

This belongs after official KNO baselines are recorded.
