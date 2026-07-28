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

RUN=ampkno_burgers_o32_allfreq_r8_ep500_seed42
CUDA_VISIBLE_DEVICES=1 nohup python -u experiments/stage4_0/train_am_pkno_burgers.py \
  --data-path "$DATA_ROOT/$BURGERS_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 64 --sub 32 \
  --ntrain 1000 --ntest 200 \
  --operator-size 32 --decompose 8 --max-modes 0 \
  --lr 1e-3 --seed 42 --save-checkpoint --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
