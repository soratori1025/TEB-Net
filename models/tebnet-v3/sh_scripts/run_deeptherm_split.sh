#!/bin/bash
set -e
CSV="/home/nvsinh1/brats_segmentation/teb-net-model/models-s3/tebnet-v3/deeptherm_data.csv"
source /home/nvsinh1/brats_segmentation/teb-net-model/.venv/bin/activate
mkdir -p logs_v3/deeptherm_split
echo "Starting DeepTherm 10-model ensemble with SPLIT models (H and SCp)..."
for batch in "0 1" "2 3" "4 5" "6 7" "8 9"; do
    for m in $batch; do
        gpu=$(( m % 2 ))
        echo "Starting member $m on GPU $gpu..."
        python scripts/train.py --seed $m --member $m --split_seed 0 --csv "$CSV" --s_prior v2_exact --gpu $gpu --overwrite --loss_weights 1 0 0 --monitor h --tag_suffix _splitH > "logs_v3/deeptherm_split/deeptherm_H_m${m}.log" 2>&1 &
        python scripts/train.py --seed $m --member $m --split_seed 0 --csv "$CSV" --s_prior v2_exact --gpu $gpu --overwrite --loss_weights 0 1.5 1 --monitor scp --tag_suffix _splitSCp > "logs_v3/deeptherm_split/deeptherm_SCp_m${m}.log" 2>&1 &
    done
    wait
done
echo "Evaluating R2 Ensemble..."
H_TAG="scratch_w1-0-0_mon-h_thermo_d5_ep100_m10_physics_sp-v2_exact_splitH"
SCP_TAG="scratch_w0-1.5-1_mon-scp_thermo_d5_ep100_m10_physics_sp-v2_exact_splitSCp"
python scripts/deeptherm_eval.py --runs benchmark_results_deeptherm/$H_TAG benchmark_results_deeptherm/$SCP_TAG --combine $H_TAG $SCP_TAG > logs_v3/deeptherm_split/deeptherm_ensemble_eval.log 2>&1
echo "Ensemble done."
