# GPU queue scripts

These scripts wait for the current PID on each GPU to exit, confirm the GPU has no compute process for several checks, then start the assigned next job.

## Current mapping

- cuda0 waits for PID 372306, then starts `ampkno_ns_v1e3_o32_allfreq_fact1_r8_t40_ep500_seed42`
- cuda1 waits for PID 364522, then starts `ampkno_burgers_o32_allfreq_r8_ep500_seed42`
- cuda2 waits for PID 1827792, then starts `pkno_shallow_water_o32_m16_r8_t40_ep500_seed42`
- cuda3 waits for PID 1851098, then starts `ampkno_ns_v1e4_o32_allfreq_fact1_r8_t40_ep500_seed42`
- cuda6 waits for PID 2457925, then starts `ampkno_shallow_water_o32_allfreq_fact1_r4_t40_ep500_seed42`

## Use on the server

Copy this `queue_gpu_jobs` directory to:

```bash
/home/lpq/Wangwanqi/PKNO_Experiments/queue_gpu_jobs
```

Then run:

```bash
cd /home/lpq/Wangwanqi/PKNO_Experiments/
chmod +x queue_gpu_jobs/*.sh
bash queue_gpu_jobs/enqueue_all.sh
```

Check queue sessions:

```bash
tmux ls | grep queue_cuda
```

Watch one waiter:

```bash
tmux attach -t queue_cuda0_after_372306
```

Detach from tmux with `Ctrl-b`, then `d`.

Check waiter logs:

```bash
tail -f logs/queue_gpu_jobs/*.wait.log
```

Check training logs after jobs start:

```bash
tail -f logs/stage4_0_am_pkno/*.log logs/stage3_0_param_kno/*.log
```

## Safety notes

The waiters poll `nvidia-smi` once per minute by default. They do not allocate GPU memory and do not create CUDA contexts. They should not affect training performance in any meaningful way.

The default `IDLE_CHECKS=5` means a job starts only after the target PID exits and the GPU appears idle for five consecutive one-minute checks.
