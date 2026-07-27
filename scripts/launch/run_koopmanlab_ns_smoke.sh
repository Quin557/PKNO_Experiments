#!/usr/bin/env bash
set -euo pipefail

source configs/data_paths.env
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -u experiments/official_kno/train_koopmanlab_ns.py \
  --koopmanlab-root "$KOOPMANLAB_ROOT" \
  --data-path "$DATA_ROOT/$NS2D_FILE" \
  --run-name smoke_koopmanlab_ns_${CUDA_VISIBLE_DEVICES} \
  --epochs 1 \
  --batch-size 10 \
  --t-in 10 \
  --t-out 40 \
  --viscosity-type 1e-3 \
  --operator-size 32 \
  --modes 16 \
  --decompose 8 \
  --device cuda
