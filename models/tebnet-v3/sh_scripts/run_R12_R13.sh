#!/bin/bash
source /home/nvsinh1/brats_segmentation/teb-net-model/.venv/bin/activate
export PYTHONPATH=$(pwd)

echo "Running R12..."
python check/nasa7_check.py \
  --h_runs benchmark_results_deeptherm/thermo_physics_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h_sp-v2_exact_splitH/random0/member*_seed* \
  --scp_runs benchmark_results_deeptherm/thermo_physics_chebyshev_3d_grouped_d5_global_w0-1.5-1_mon-scp_sp-v2_exact_splitSCp/random0/member*_seed* \
  --complete_high --write_dat logs_v3/c5_random0.dat > logs_v3/R12_nasa7.log 2>&1

echo "Running R13 step 1..."
python check/nasa7_check.py \
  --h_runs benchmark_results_deeptherm/thermo_physics_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h_sp-v2_exact_splitH/random0/member*_seed* \
  --scp_runs benchmark_results_deeptherm/thermo_physics_chebyshev_3d_grouped_d5_global_w0-1.5-1_mon-scp_sp-v2_exact_splitSCp/random0/member*_seed* \
  --smiles_csv rmg_cp_reference.csv \
  --complete_high --write_dat logs_v3/c5_rmg.dat > logs_v3/R13_nasa7_rmg.log 2>&1

echo "Running R13 step 2..."
python check/high_t_validation.py logs_v3/c5_rmg.dat \
  --reference rmg_cp_reference.csv \
  --out_csv logs_v3/high_t_validation.csv > logs_v3/R13_high_t.log 2>&1

echo "Done."
