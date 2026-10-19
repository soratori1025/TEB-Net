#!/usr/bin/env bash
source ../../.venv/bin/activate
export PYTHONPATH=/home/nvsinh1/brats_segmentation/teb-net-model/models-s3/tebnet-v1:/home/nvsinh1/brats_segmentation/teb-net-model
export CUDA_VISIBLE_DEVICES=0
python baselines_ml.py --csv ../deeptherm_data.csv --model xgb --features physics+ecfp --split_type random --split_seed 2 --out_dir benchmark_results_baselines --xgb_device cuda
