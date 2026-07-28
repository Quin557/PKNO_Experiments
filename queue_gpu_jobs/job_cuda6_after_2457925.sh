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

RUN=ampkno_shallow_water_o32_allfreq_fact1_r4_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=6 nohup python -u experiments/stage4_0/train_am_pkno_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 5 --t-in 10 --t-out 40 \
  --ntrain 900 --ntest 100 --dt 0.01 \
  --operator-size 32 --decompose 4 --max-modes 0 \
  --operator-factorization factorized --factorized-rank 1 \
  --output-scale 0.005 --lr 5e-5 --max-grad-norm 0.1 \
  --seed 42 --save-checkpoint --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
