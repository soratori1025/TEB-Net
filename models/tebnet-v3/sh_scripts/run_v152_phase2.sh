#!/bin/bash
set -e
OUT=benchmark_results_v152
H="--loss_weights 1 0 0 --monitor h"
SCP="--loss_weights 0 1.5 1 --monitor scp"
CSV="/home/nvsinh1/brats_segmentation/teb-net-model/models-baseline/deeptherm_data.csv"
echo "Running random splits 2, 3, 4..."
for s in 2 3 4; do
  python scripts/train.py --csv "$CSV" --seed 0 --split_seed $s $H --out_dir $OUT > logs/v152_H_$s.log 2>&1 &
  python scripts/train.py --csv "$CSV" --seed 0 --split_seed $s $SCP --out_dir $OUT > logs/v152_SCp_$s.log 2>&1 &
done
wait
echo "Random splits finished."
echo "Running functional group splits..."
for g in "Ether group" "Epoxide group" "Enol group"; do
  g_log=${g// /_}
  python scripts/train.py --csv "$CSV" --seed 0 --split_type functional_group --test_group "$g" $H --out_dir $OUT > "logs/fg_H_${g_log}.log" 2>&1 &
  python scripts/train.py --csv "$CSV" --seed 0 --split_type functional_group --test_group "$g" $SCP --out_dir $OUT > "logs/fg_SCp_${g_log}.log" 2>&1 &
done
wait
echo "Functional group splits finished."
python scripts/deeptherm_eval.py --runs $OUT \
  --combine thermo_physics_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h \
            thermo_physics_chebyshev_3d_grouped_d5_global_w0-1.5-1_mon-scp > evaluation_output_phase2.txt
