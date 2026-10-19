#!/bin/bash
set -e
export PYTHONPATH=/home/nvsinh1/brats_segmentation/teb-net-model/models-s3/tebnet-v3
CSV=/home/nvsinh1/brats_segmentation/teb-net-model/models-baseline/sota/bruce_benchmark/data/wudily_cho_processed.csv

mkdir -p logs_v3

echo "Computing physics features..."
python scripts/compute_physics_features.py --csv $CSV

# Run folds in parallel on 2 GPUs
for k in 0 1 2 3 4; do
    echo "Running fold $k..."
    CUDA_VISIBLE_DEVICES=0 python scripts/train_wudily_cv.py --csv $CSV --folds $k --loss_weights 1 0 0 --monitor h --s_prior v2_exact --tag_suffix _v2exact > logs_v3/wudily_H_$k.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python scripts/train_wudily_cv.py --csv $CSV --folds $k --loss_weights 0 1.5 1 --monitor scp --s_prior v2_exact --tag_suffix _v2exact > logs_v3/wudily_SCp_$k.log 2>&1 &
    wait
done

echo "Running prior_only..."
CUDA_VISIBLE_DEVICES=0 python scripts/train_wudily_cv.py --csv $CSV --prior_only --s_prior v2_exact --tag_suffix _v2exact > logs_v3/wudily_prior.log 2>&1

echo "Summarizing..."
H_TAG="scratch_w1-0-0_mon-h_thermo_d5_ep100_m1_physics_k5_cv0_s0_v2exact"
SCP_TAG="scratch_w0-1.5-1_mon-scp_thermo_d5_ep100_m1_physics_k5_cv0_s0_v2exact"
python scripts/train_wudily_cv.py --csv $CSV --summarize --combine $H_TAG $SCP_TAG > logs_v3/wudily_summary.log 2>&1

echo "All done."
