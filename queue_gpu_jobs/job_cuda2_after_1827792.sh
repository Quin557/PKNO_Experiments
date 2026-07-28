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
STAGE=stage3_0_param_kno
LOG_DIR="logs/$STAGE"
OUT_DIR="outputs/$STAGE"
mkdir -p "$LOG_DIR" "$OUT_DIR" "results/$STAGE" "reports/$STAGE"

RUN=pkno_shallow_water_o32_m16_r8_t40_ep500_seed42
CUDA_VISIBLE_DEVICES=2 nohup python -u experiments/stage3_0/train_pkno_shallow_water.py \
  --data-path "$DATA_ROOT/$SHALLOW_WATER_FILE" \
  --run-name "$RUN" --output-dir "$OUT_DIR" \
  --epochs 500 --batch-size 5 --t-in 10 --t-out 40 \
  --ntrain 900 --ntest 100 \
  --operator-size 32 --modes 16 --decompose 8 \
  --dt 0.01 --lr 2e-4 --delta-scale 0.02 --max-grad-norm 0.5 \
  --seed 42 --save-checkpoint --device cuda \
  > "$LOG_DIR/$RUN.log" 2>&1 &
echo $!
