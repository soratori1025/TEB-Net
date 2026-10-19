#!/bin/bash
set -e
source ../../../.venv/bin/activate

CSV="/home/nvsinh1/brats_segmentation/teb-net-model/models-baseline/deeptherm_data.csv"
SPLITS="splits_random.json"
OUT_DIR="chemprop_dmpnn_random_h"

chemprop train --data-path "$CSV" --task-type regression --smiles-columns smiles \
  --target-columns 'Hf_298.15K_kcal/mol' \
  --splits-file "$SPLITS" --epochs 100 --save-dir "$OUT_DIR"

python evaluate_chemprop_deeptherm_h.py > evaluation_h_output.txt
