#!/bin/bash
source /home/nvsinh1/brats_segmentation/teb-net-model/.venv/bin/activate
export PYTHONPATH=$(pwd)
echo "Running nasa7_check.py" > log.txt
python check/nasa7_check.py --h_runs benchmark_results_deeptherm/thermo_physics_chebyshev_3d_grouped_d5_global_sp-v2_exact/random0/* --smiles_csv rmg_cp_reference.csv --complete_high --write_dat c5_rmg.dat >> log.txt 2>&1
echo "Running high_t_validation.py" >> log.txt
python check/high_t_validation.py c5_rmg.dat --reference rmg_cp_reference.csv --deeptherm_csv ../../models-baseline/deeptherm_data.csv > high_t_validation.txt 2>> log.txt
echo "Running plot_cp_completion.py" >> log.txt
python check/plot_cp_completion.py c5_rmg.dat --reference rmg_cp_reference.csv --error_panel --out fig_cp_completion >> log.txt 2>&1
echo "Done" >> log.txt
