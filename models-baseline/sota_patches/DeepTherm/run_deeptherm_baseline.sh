#!/bin/bash
set -e

CSV="../../deeptherm_data.csv"
TARGETS="Hf_298.15K_kcal/mol S_298.15K_cal/(molK) cp_300K_cal/(molK) cp_400K_cal/(molK) cp_500K_cal/(molK) cp_600K_cal/(molK) cp_800K_cal/(molK) cp_1000K_cal/(molK) cp_1500K_cal/(molK)"
PRETRAINED="runs/qm9_new/model_00.pt"

# ── 1. DeepTherm retrained, NO pretrain, single model, 5 random seeds ──
echo "=== DeepTherm NO pretrain, single model, random splits ==="
for s in 0 1 2 3 4; do
  echo "  seed $s ..."
  python train.py finetune \
    --data "$CSV" --smiles-column smiles \
    --targets $TARGETS \
    --save-dir runs/deeptherm_nopretrain_random_s${s} \
    --descriptor morgan \
    --ensemble-size 1 --seed $s \
    --epochs 100 --patience 15
done

# ── 2. DeepTherm WITH pretrain (QM9), single model, 5 random seeds ──
echo "=== DeepTherm WITH pretrain, single model, random splits ==="
for s in 0 1 2 3 4; do
  echo "  seed $s ..."
  python train.py finetune \
    --data "$CSV" --smiles-column smiles \
    --targets $TARGETS \
    --pretrained "$PRETRAINED" \
    --save-dir runs/deeptherm_pretrain_random_s${s} \
    --descriptor morgan \
    --ensemble-size 1 --seed $s \
    --epochs 100 --patience 15
done

echo "Done."
