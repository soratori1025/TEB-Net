#!/bin/bash
set -e
OUT=benchmark_results_v153
H="--csv ../../models-baseline/deeptherm_data.csv --loss_weights 1 0 0 --monitor h"
SCP="--csv ../../models-baseline/deeptherm_data.csv --loss_weights 0 1.5 1 --monitor scp"

echo "Running v15.3..."
for s in 0 1; do
    PYTHONPATH=/home/nvsinh1/brats_segmentation/teb-net-model python scripts/train.py --seed 0 --split_seed $s $H --out_dir $OUT > logs/v153_H_$s.log 2>&1 &
    PYTHONPATH=/home/nvsinh1/brats_segmentation/teb-net-model python scripts/train.py --seed 0 --split_seed $s $SCP --out_dir $OUT > logs/v153_SCp_$s.log 2>&1 &
done

for seed in 0 1; do
    PYTHONPATH=/home/nvsinh1/brats_segmentation/teb-net-model python scripts/train.py --seed $seed --split_type complexity $H --out_dir $OUT > logs/v153_Hcx_$seed.log 2>&1 &
    PYTHONPATH=/home/nvsinh1/brats_segmentation/teb-net-model python scripts/train.py --seed $seed --split_type complexity $SCP --out_dir $OUT > logs/v153_SCpcx_$seed.log 2>&1 &
done

wait
echo "Done."
