#!/bin/bash
set -e
export PYTHONPATH=/home/nvsinh1/brats_segmentation/teb-net-model
CSV="../../models-baseline/deeptherm_data.csv"
OUT="benchmark_results_v153"

for m in {1..9}; do
  echo "Training member $m..."
  python scripts/train.py --csv "$CSV" --seed 0 --split_type random --split_seed 0 --member $m --loss_weights 1 0 0 --monitor h --out_dir "$OUT" > "logs/v153_ens_H_$m.log" 2>&1 &
  python scripts/train.py --csv "$CSV" --seed 0 --split_type random --split_seed 0 --member $m --loss_weights 0 1.5 1 --monitor scp --out_dir "$OUT" > "logs/v153_ens_SCp_$m.log" 2>&1 &
  wait
done
echo "Done."
