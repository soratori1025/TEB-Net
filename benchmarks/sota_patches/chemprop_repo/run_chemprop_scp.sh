#!/bin/bash
set -e
source ../../../.venv/bin/activate
CSV="/home/nvsinh1/brats_segmentation/teb-net-model/models-baseline/deeptherm_data.csv"
SPLITS="splits_random.json"

echo "Running Chemprop for S..."
chemprop train --data-path "$CSV" --task-type regression --smiles-columns smiles \
  --target-columns 'S_298.15K_cal/(molK)' \
  --splits-file "$SPLITS" --epochs 100 --save-dir "chemprop_dmpnn_random_s"

echo "Running Chemprop for Cp..."
chemprop train --data-path "$CSV" --task-type regression --smiles-columns smiles \
  --target-columns 'cp_300K_cal/(molK)' 'cp_400K_cal/(molK)' 'cp_500K_cal/(molK)' 'cp_600K_cal/(molK)' 'cp_800K_cal/(molK)' 'cp_1000K_cal/(molK)' 'cp_1500K_cal/(molK)' \
  --splits-file "$SPLITS" --epochs 100 --save-dir "chemprop_dmpnn_random_cp"

python evaluate_chemprop_deeptherm_scp.py > evaluation_scp_output.txt
