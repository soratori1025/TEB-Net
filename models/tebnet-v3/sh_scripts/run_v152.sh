#!/bin/bash
set -e
mkdir -p logs

CSV="/home/nvsinh1/brats_segmentation/teb-net-model/models-baseline/deeptherm_data.csv"
python scripts/compute_physics_features.py --csv "$CSV" --overwrite

OUT=benchmark_results_v152
H="--loss_weights 1 0 0 --monitor h"
SCP="--loss_weights 0 1.5 1 --monitor scp"

echo "Running TEB-Net training jobs in background..."

for s in 0 1; do
  python scripts/train.py --csv "$CSV" --seed 0 --split_seed $s $H --out_dir $OUT > logs/v152_H_$s.log 2>&1 &
  python scripts/train.py --csv "$CSV" --seed 0 --split_seed $s $SCP --out_dir $OUT > logs/v152_SCp_$s.log 2>&1 &
done

for seed in 0 1; do
  python scripts/train.py --csv "$CSV" --seed $seed --split_type complexity $H --out_dir $OUT > logs/v152_Hcx_$seed.log 2>&1 &
  python scripts/train.py --csv "$CSV" --seed $seed --split_type complexity $SCP --out_dir $OUT > logs/v152_SCpcx_$seed.log 2>&1 &
done

wait
echo "All training jobs finished. Evaluating..."

python scripts/deeptherm_eval.py --runs $OUT \
  --combine thermo_physics_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h \
            thermo_physics_chebyshev_3d_grouped_d5_global_w0-1.5-1_mon-scp > evaluation_output.txt

cat evaluation_output.txt
