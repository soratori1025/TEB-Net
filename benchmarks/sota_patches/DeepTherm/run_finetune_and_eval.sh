#!/bin/bash
set -e
echo "Starting finetuning..."
/home/nvsinh1/brats_segmentation/teb-net-model/.venv/bin/python train.py finetune \
  --data data/thermo_train_val.csv \
  --targets Hf_298 S_298 Cp_300 Cp_400 Cp_500 Cp_600 Cp_800 Cp_1000 Cp_1500 \
  --pretrained runs/qm9_new/model_00.pt \
  --save-dir runs/deeptherm_new \
  --descriptor morgan \
  --ensemble-size 10

echo "Finetuning completed. Starting prediction..."
/home/nvsinh1/brats_segmentation/teb-net-model/.venv/bin/python predict.py \
  --data data/thermo_test.csv \
  --checkpoint-dir runs/deeptherm_new \
  --output predictions_new.csv

echo "Prediction completed. Evaluating..."
# Update evaluate_predictions.py to read predictions_new.csv
sed -i 's/predictions.csv/predictions_new.csv/g' evaluate_predictions.py
/home/nvsinh1/brats_segmentation/teb-net-model/.venv/bin/python evaluate_predictions.py > final_evaluation_results.txt
echo "Done!"
