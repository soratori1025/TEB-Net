#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=pilot_sweep
rm -rf "$OUT"
mkdir -p "$OUT"
run_one() {
  local lam=$1 gpu=$2
  echo "[start] lambda=$lam gpu=$gpu"
  python -u scripts/train.py --seed 0 --member 0 --split_seed 0 \
    --csv deeptherm_data.csv --s_prior v2_exact \
    --homolog_weight "$lam" --max_epochs 10 \
    --gpu "$gpu" --out_dir "$OUT" --overwrite \
    > "/tmp/sweep_lam${lam}.log" 2>&1
  echo "[done ] lambda=$lam exit=$?"
}
run_one 1e-3 0 &
run_one 3e-3 1 &
wait
run_one 1e-2 0 &
wait
echo "SWEEP_COMPLETE"
