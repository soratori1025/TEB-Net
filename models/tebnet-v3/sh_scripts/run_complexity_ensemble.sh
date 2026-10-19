#!/bin/bash
set -e
export PYTHONPATH=$(pwd)
source /home/nvsinh1/brats_segmentation/teb-net-model/.venv/bin/activate
mkdir -p logs_v3

CSV="/home/nvsinh1/brats_segmentation/teb-net-model/models-s3/tebnet-v3/deeptherm_data.csv"

echo "Starting DeepTherm Complexity Split 10-model ensemble (2 concurrent jobs, 1 per GPU)..."
for batch in "0 1" "2 3" "4 5" "6 7" "8 9"; do
    for m in $batch; do
        gpu=$(( m % 2 ))
        echo "Starting member $m on GPU $gpu..."
        python scripts/train.py --seed $m --member $m --split_seed 0 --split_type complexity --csv "$CSV" --s_prior v2_exact --gpu $gpu --overwrite > "logs_v3/deeptherm_complexity_m${m}.log" 2>&1 &
    done
    wait
done

echo "Evaluating ensemble..."
python scripts/deeptherm_eval.py --runs benchmark_results_deeptherm/thermo_physics_chebyshev_3d_grouped_d5_global_sp-v2_exact > logs_v3/deeptherm_complexity_ensemble_eval.log 2>&1
echo "Ensemble done."
