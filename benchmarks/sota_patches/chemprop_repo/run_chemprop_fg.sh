#!/bin/bash
set -e

CSV="../../deeptherm_data.csv"

for g in "Ether group" "Epoxide group" "Enol group"; do
  gname=${g// /_}
  SPLITS="splits_fg_${gname}.json"
  python export_splits.py --csv "$CSV" --split_type functional_group --test_group "$g" --out "$SPLITS"
  
  OUT_DIR_H="chemprop_fg_h_${gname}"
  chemprop train --data-path "$CSV" --task-type regression --smiles-columns smiles \
    --target-columns 'Hf_298.15K_kcal/mol' \
    --splits-file "$SPLITS" --epochs 100 --save-dir "$OUT_DIR_H" > "log_fg_h_${gname}.txt" 2>&1
    
  OUT_DIR_SCP="chemprop_fg_scp_${gname}"
  cp_cols="cp_300K_cal/(molK) cp_400K_cal/(molK) cp_500K_cal/(molK) cp_600K_cal/(molK) cp_800K_cal/(molK) cp_1000K_cal/(molK) cp_1500K_cal/(molK)"
  chemprop train --data-path "$CSV" --task-type regression --smiles-columns smiles \
    --target-columns 'S_298.15K_cal/(molK)' $cp_cols \
    --splits-file "$SPLITS" --epochs 100 --save-dir "$OUT_DIR_SCP" > "log_fg_scp_${gname}.txt" 2>&1
    
  python evaluate_chemprop_fg.py "$g" > "evaluation_fg_${gname}.txt"
done
