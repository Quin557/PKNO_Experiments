#!/usr/bin/env bash
set -euo pipefail

source configs/data_paths.env
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STAGE="${STAGE:-stage0_kno_baseline}"
OUT_DIR="${OUT_DIR:-outputs/$STAGE}"
mkdir -p "$OUT_DIR"

python -u experiments/official_kno/train_koopmanlab_ns.py \
  --koopmanlab-root "$KOOPMANLAB_ROOT" \
  --data-path "$DATA_ROOT/$NS2D_FILE" \
  --run-name smoke_koopmanlab_ns_${CUDA_VISIBLE_DEVICES} \
  --output-dir "$OUT_DIR" \
  --epochs 1 \
  --batch-size 10 \
  --t-in 10 \
  --t-out 40 \
  --viscosity-type 1e-3 \
  --operator-size 32 \
  --modes 16 \
  --decompose 8 \
  --device cuda
