#!/usr/bin/env bash
set -e
cd /home/nvsinh1/brats_segmentation/teb-net-model/models-baseline/sota/chemprop_repo
source ../../../.venv/bin/activate

CSV=/home/nvsinh1/brats_segmentation/teb-net-model/models-baseline/deeptherm_data.csv
SPLITS=/home/nvsinh1/brats_segmentation/teb-net-model/models-baseline/sota/chemprop_repo/splits_random.json
OUT_DIR=chemprop_dmpnn_random

echo "Bắt đầu train Chemprop D-MPNN thuần trên 5 split random của DeepTherm..."

chemprop train --data-path "$CSV" --task-type regression --smiles-columns smiles \
  --target-columns 'Hf_298.15K_kcal/mol' 'S_298.15K_cal/(molK)' 'cp_300K_cal/(molK)' 'cp_400K_cal/(molK)' 'cp_500K_cal/(molK)' 'cp_600K_cal/(molK)' 'cp_800K_cal/(molK)' 'cp_1000K_cal/(molK)' 'cp_1500K_cal/(molK)' \
  --splits-file "$SPLITS" \
  --epochs 100 \
  --save-dir "$OUT_DIR"

echo "Chạy xong Chemprop!"
