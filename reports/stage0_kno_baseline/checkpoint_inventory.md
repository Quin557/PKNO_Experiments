# Stage 0 KNO Checkpoint Inventory

本文件只登记真正可用于 evaluation-only 的 checkpoint。`time_error.pt`、误差张量和 `metrics.csv` 不计入 checkpoint。

扫描目录：`outputs\stage0_kno_baseline`

| task | run name | checkpoint path | checkpoint epoch | loadable | status | args/config |
|---|---|---|---:|---|---|---|
| Burgers | `kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42` | `` |  | no | missing | outputs\stage0_kno_baseline\kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42\args.json<br>outputs\stage0_kno_baseline\kno_koopmanlab_burgers_o32_m16_r8_ep500_seed42\config.yaml |
| Navier-Stokes v1e-3 | `kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_lr001_seed42_rerun1` | `` |  | no | missing | outputs\stage0_kno_baseline\kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_lr001_seed42_rerun1\args.json<br>outputs\stage0_kno_baseline\kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_lr001_seed42_rerun1\config.yaml |
| Navier-Stokes v1e-3 | `kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_seed42` | `` |  | no | missing | outputs\stage0_kno_baseline\kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_seed42\args.json<br>outputs\stage0_kno_baseline\kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_seed42\config.yaml |
| Navier-Stokes v1e-4 | `kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_lr001_seed42_rerun1` | `` |  | no | missing | outputs\stage0_kno_baseline\kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_lr001_seed42_rerun1\args.json<br>outputs\stage0_kno_baseline\kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_lr001_seed42_rerun1\config.yaml |
| Navier-Stokes v1e-4 | `kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_seed42` | `` |  | no | missing | outputs\stage0_kno_baseline\kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_seed42\args.json<br>outputs\stage0_kno_baseline\kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_seed42\config.yaml |
| Shallow Water | `kno_koopmanlab_shallow_water_o32_m16_r8_t40_ep500_seed42` | `` |  | no | missing | outputs\stage0_kno_baseline\kno_koopmanlab_shallow_water_o32_m16_r8_t40_ep500_seed42\args.json<br>outputs\stage0_kno_baseline\kno_koopmanlab_shallow_water_o32_m16_r8_t40_ep500_seed42\config.yaml |

## 结论

- `loadable=yes` 才能进入独立评估脚本。
- `missing_model_state_dict` 表示文件即使存在也不能算作模型 checkpoint。
- 在 checkpoint 状态确认前，不应启动长时间重训。
