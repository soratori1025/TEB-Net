#!/bin/bash
set -e
export PYTHONPATH=$(pwd)
source /home/nvsinh1/brats_segmentation/teb-net-model/.venv/bin/activate
mkdir -p logs_v3

echo "Precomputing caches for the 3 Ureel datasets..."
for CSV in "carbenium_hc.csv" "radical_hc.csv" "stable_hc.csv"; do
    python scripts/train_cv.py --csv "/home/nvsinh1/brats_segmentation/teb-net-model/data/splits/$CSV" --prior_only --s_prior v2_exact --h_unit j > "logs_v3/ureel_prior_${CSV}.log" 2>&1
done

echo "Starting Ureel Nested CV (10x9) on 3 datasets (Joint models, 2 outer folds concurrently)..."
run_dataset() {
    CSV=$1
    NAME=$2
    echo "Processing $NAME..."
    for batch in "0 1" "2 3" "4 5" "6 7" "8 9"; do
        for k in $batch; do
            gpu=$(( k % 2 ))
            echo "Starting outer fold $k on GPU $gpu..."
            python scripts/train_cv.py --csv "$CSV" --outer_folds $k --s_prior v2_exact --h_unit j --gpu $gpu > "logs_v3/ureel_${NAME}_k${k}.log" 2>&1 &
        done
        wait
    done
    python scripts/train_cv.py --csv "$CSV" --s_prior v2_exact --h_unit j --summarize > "logs_v3/ureel_${NAME}_summary.log" 2>&1
}

run_dataset "/home/nvsinh1/brats_segmentation/teb-net-model/data/splits/carbenium_hc.csv" "carbenium"
run_dataset "/home/nvsinh1/brats_segmentation/teb-net-model/data/splits/radical_hc.csv" "radical"
run_dataset "/home/nvsinh1/brats_segmentation/teb-net-model/data/splits/stable_hc.csv" "stable"

echo "All Ureel Nested CVs completed."
