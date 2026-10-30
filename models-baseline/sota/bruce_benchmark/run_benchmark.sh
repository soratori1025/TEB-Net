#!/bin/bash
set -e
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
PREP_ONLY=false
EVAL_ONLY=false
QUICK=false
MAX_EPOCHS=200
PATIENCE=40
GPUS="0,1"
for arg in "$@"; do
    case $arg in
        --prep-only) PREP_ONLY=true ;;
        --eval-only) EVAL_ONLY=true ;;
        --quick) QUICK=true; MAX_EPOCHS=50; PATIENCE=15 ;;
        --gpus=*) GPUS="${arg#*=}" ;;
        --epochs=*) MAX_EPOCHS="${arg#*=}" ;;
    esac
done
echo "============================================================"
echo " Bruce Benchmark — TEB-Net v14.2"
echo "============================================================"
echo " Python:     $PYTHON"
echo " GPUs:       $GPUS"
echo " Max Epochs: $MAX_EPOCHS"
echo " Patience:   $PATIENCE"
echo " Quick mode: $QUICK"
echo "============================================================"
echo ""
if [ "$EVAL_ONLY" = false ]; then
    echo "[Step 1/3] Preparing WUDILY-CHO dataset..."
    $PYTHON prepare_dataset.py
    echo ""
fi
if [ "$PREP_ONLY" = false ] && [ "$EVAL_ONLY" = false ]; then
    echo "[Step 2/3] Training TEB-Net v14.2 (5 seeds)..."
    if [ "$QUICK" = true ]; then
        echo "  (Quick mode: training seed 42 only)"
        $PYTHON train_tebnet.py \
            --seed 42 --split_seed 42 \
            --max_epochs $MAX_EPOCHS --patience $PATIENCE \
            --gpu 0
    else
        $PYTHON run_all_seeds.py \
            --max_epochs $MAX_EPOCHS \
            --patience $PATIENCE \
            --gpus "$GPUS"
    fi
    echo ""
fi
if [ "$PREP_ONLY" = false ]; then
    echo "[Step 3/3] Generating evaluation report..."
    $PYTHON evaluate.py
    echo ""
fi
echo "============================================================"
echo " Benchmark complete!"
echo " Report: results/benchmark_report.md"
echo "============================================================"
