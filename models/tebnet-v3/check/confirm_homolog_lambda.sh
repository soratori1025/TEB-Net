#!/usr/bin/env bash
# Full-length (100 epoch) confirmation of the two candidate weights before committing
# to the 10-member ensemble.
#
# The 10-epoch sweep put lambda=1e-3 on par with the unregularised baseline
# (val dH298 MAE 8759 vs 8847 J/mol at epoch 8) and lambda=1e-2 clearly behind
# (10637), with lambda=3e-3 in between (8902).  A 10-epoch comparison is not the
# operating point though -- the run is 100 epochs with a cosine schedule -- so both
# candidates get a full-length run on the canonical split (member 0).
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=confirm_homolog
rm -rf "$OUT"
mkdir -p "$OUT"

run_one() {  # $1=lambda  $2=gpu
  local lam=$1 gpu=$2
  echo "[start] lambda=$lam gpu=$gpu $(date +%H:%M:%S)"
  python -u scripts/train.py --seed 0 --member 0 --split_seed 0 \
    --csv deeptherm_data.csv --s_prior v2_exact \
    --homolog_weight "$lam" --max_epochs 100 \
    --gpu "$gpu" --out_dir "$OUT" --overwrite \
    > "check/confirm_lam${lam}.log" 2>&1
  echo "[done ] lambda=$lam exit=$? $(date +%H:%M:%S)"
}

run_one 1e-3 0 &
run_one 3e-3 1 &
wait
echo "CONFIRM_COMPLETE"
