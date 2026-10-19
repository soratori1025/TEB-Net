#!/usr/bin/env bash
set -e

export PYTHONPATH=/home/nvsinh1/brats_segmentation/teb-net-model
CSV="../../models-baseline/deeptherm_data.csv"
OUT="benchmark_results_v153"

for s in 0 1 2 3 4; do
  python scripts/train.py --csv $CSV --seed 0 --split_type random --split_seed $s --monitor deeptherm --out_dir $OUT > logs/v153_multi_$s.log 2>&1 &
done

wait
echo "Done."
