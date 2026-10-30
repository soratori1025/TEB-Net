#!/bin/bash
export PYTHONPATH=$(pwd)
source /home/nvsinh1/brats_segmentation/teb-net-model/.venv/bin/activate
mkdir -p logs_v3/evaluations
echo "Evaluating DeepTherm Random Splits (R1, R2)..."
python scripts/deeptherm_eval.py --runs benchmark_results_deeptherm/scratch_w1-0-0_mon-h_thermo_d5_ep100_m*_physics_sp-v2_exact_splitH \
    benchmark_results_deeptherm/scratch_w0-1.5-1_mon-scp_thermo_d5_ep100_m*_physics_sp-v2_exact_splitSCp \
    --combine scratch_w1-0-0_mon-h_thermo_d5_ep100_m10_physics_sp-v2_exact_splitH \
              scratch_w0-1.5-1_mon-scp_thermo_d5_ep100_m10_physics_sp-v2_exact_splitSCp > logs_v3/evaluations/R12_eval.log 2>&1
python scripts/deeptherm_eval.py --runs benchmark_results_deeptherm > logs_v3/evaluations/all_deeptherm.log 2>&1
echo "Extraction scripts setup."
