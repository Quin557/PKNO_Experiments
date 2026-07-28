#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/lpq/Wangwanqi/PKNO_Experiments/

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -d "$HOME/miniconda3" ]]; then
  CONDA_BASE="$HOME/miniconda3"
elif [[ -d "$HOME/anaconda3" ]]; then
  CONDA_BASE="$HOME/anaconda3"
else
  echo "ERROR: conda was not found" >&2
  exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate pkno-exp

source configs/data_paths.env
STAGE=stage4_0_am_pkno
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

RUN=ampkno_ns_v1e3_o32_allfreq_fact1_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/stage4_0/train_am_pkno_ns_v1e3.py \
  --data-path "$DATA_ROOT/$NS2D_V1E3_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 10 --t-in 10 --t-out 40 \
  --ntrain 1000 --ntest 200 \
  --operator-size 32 --decompose 8 --max-modes 0 \
  --operator-factorization factorized --factorized-rank 1 \
  --output-scale 0.015 --lr 5e-4 --max-grad-norm 1.0 \
  --seed 42 --save-checkpoint --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
