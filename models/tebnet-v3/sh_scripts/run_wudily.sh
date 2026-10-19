#!/bin/bash
set -e
export PYTHONPATH=/home/nvsinh1/brats_segmentation/teb-net-model/models-s3/tebnet-v1
CSV=/home/nvsinh1/brats_segmentation/teb-net-model/models-baseline/sota/bruce_benchmark/data/wudily_cho_processed.csv

echo "Computing physics features..."
python scripts/compute_physics_features.py --csv $CSV

for k in 0 1 2 3 4; do
    echo "Running fold $k..."
    python scripts/train_wudily_cv.py --csv $CSV --folds $k --loss_weights 1 0 0 --monitor h > logs/wudily_H_$k.log 2>&1 &
    python scripts/train_wudily_cv.py --csv $CSV --folds $k --loss_weights 0 1.5 1 --monitor scp > logs/wudily_SCp_$k.log 2>&1 &
    wait
done

echo "Running prior_only..."
python scripts/train_wudily_cv.py --csv $CSV --prior_only > logs/wudily_prior.log 2>&1

echo "Summarizing..."
H_TAG="scratch_w1-0-0_mon-h_thermo_d5_ep100_m1_physics_k5_cv0_s0"
SCP_TAG="scratch_w0-1.5-1_mon-scp_thermo_d5_ep100_m1_physics_k5_cv0_s0"
python scripts/train_wudily_cv.py --csv $CSV --summarize --combine $H_TAG $SCP_TAG > logs/wudily_summary.log 2>&1

echo "All done."
