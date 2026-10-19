#!/usr/bin/env bash
# Sweep the CH2-homolog weight after the batch-size bug fix.
#
# Rationale: with the reduction bug fixed the penalty no longer scales with the batch
# size, but at lambda=0.1 it is still ~2.9x the per-molecule loss, i.e. it still
# dominates.  A prior should perturb the objective, not replace it: the penalty wants
# to sit at roughly 5-10% of the main loss, which puts lambda near 1e-3 .. 1e-2.
#
# Two workers on two GPUs; each run is 10 epochs (~6 min).
set -u
cd "$(dirname "$0")/.." || exit 1
OUT=pilot_sweep
rm -rf "$OUT"
mkdir -p "$OUT"

run_one() {  # $1=lambda  $2=gpu
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
