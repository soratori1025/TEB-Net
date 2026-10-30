#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
H_RUNS=(benchmark_results_deeptherm/thermo_physics_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h_sp-v2_exact_splitH/random0/member*_seed*)
SCP_RUNS=(benchmark_results_deeptherm/thermo_physics_chebyshev_3d_grouped_d5_global_w0-1.5-1_mon-scp_sp-v2_exact_splitSCp/random0/member*_seed*)
echo "Running R12..."
"$PYTHON" check/nasa7_check.py \
  --h_runs "${H_RUNS[@]}" \
  --scp_runs "${SCP_RUNS[@]}" \
  --complete_high --agg trimmed --write_dat logs_v3/c5_random0.dat > logs_v3/R12_nasa7.log 2>&1
echo "Running R13 step 1..."
"$PYTHON" check/nasa7_check.py \
  --h_runs "${H_RUNS[@]}" \
  --scp_runs "${SCP_RUNS[@]}" \
  --smiles_csv rmg_cp_reference.csv \
  --complete_high --agg trimmed --write_dat logs_v3/c5_rmg.dat > logs_v3/R13_nasa7_rmg.log 2>&1
echo "Running R13 step 2..."
"$PYTHON" check/high_t_validation.py logs_v3/c5_rmg.dat \
  --reference rmg_cp_reference.csv \
  --as_trained logs_v3/c5_rmg.as_trained.dat \
  --out_csv high_t_validation.csv > logs_v3/R13_high_t.log 2>&1
echo "Done."
