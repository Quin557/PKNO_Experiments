#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/lpq/Wangwanqi/PKNO_Experiments/

mkdir -p logs/queue_gpu_jobs

tmux new-session -d -s queue_cuda0_after_372306 \
  'cd /home/lpq/Wangwanqi/PKNO_Experiments/ && INTERVAL=60 IDLE_CHECKS=5 bash queue_gpu_jobs/wait_then_run.sh 0 372306 kno_koopmanlab_ns_v1e4_o32_m16_r8_t40_ep500_lr001_seed42_rerun1 queue_gpu_jobs/job_cuda0_after_372306.sh 2>&1 | tee -a logs/queue_gpu_jobs/cuda0_after_372306.wait.log'

tmux new-session -d -s queue_cuda1_after_364522 \
  'cd /home/lpq/Wangwanqi/PKNO_Experiments/ && INTERVAL=60 IDLE_CHECKS=5 bash queue_gpu_jobs/wait_then_run.sh 1 364522 kno_koopmanlab_ns_v1e3_o32_m16_r8_t40_ep500_lr001_seed42_rerun1 queue_gpu_jobs/job_cuda1_after_364522.sh 2>&1 | tee -a logs/queue_gpu_jobs/cuda1_after_364522.wait.log'

tmux new-session -d -s queue_cuda2_after_1827792 \
  'cd /home/lpq/Wangwanqi/PKNO_Experiments/ && INTERVAL=60 IDLE_CHECKS=5 bash queue_gpu_jobs/wait_then_run.sh 2 1827792 kno_koopmanlab_shallow_water_o32_m16_r8_t40_ep500_seed42 queue_gpu_jobs/job_cuda2_after_1827792.sh 2>&1 | tee -a logs/queue_gpu_jobs/cuda2_after_1827792.wait.log'

tmux new-session -d -s queue_cuda3_after_1851098 \
  'cd /home/lpq/Wangwanqi/PKNO_Experiments/ && INTERVAL=60 IDLE_CHECKS=5 bash queue_gpu_jobs/wait_then_run.sh 3 1851098 pkno_ns_v1e3_o32_m16_r8_t40_ep500_seed42 queue_gpu_jobs/job_cuda3_after_1851098.sh 2>&1 | tee -a logs/queue_gpu_jobs/cuda3_after_1851098.wait.log'

tmux new-session -d -s queue_cuda6_after_2457925 \
  'cd /home/lpq/Wangwanqi/PKNO_Experiments/ && INTERVAL=60 IDLE_CHECKS=5 bash queue_gpu_jobs/wait_then_run.sh 6 2457925 amkno_ns_v1e3_o32_allfreq_fact1_r8_t40_ep500_seed42 queue_gpu_jobs/job_cuda6_after_2457925.sh 2>&1 | tee -a logs/queue_gpu_jobs/cuda6_after_2457925.wait.log'

tmux ls | grep queue_cuda || true
