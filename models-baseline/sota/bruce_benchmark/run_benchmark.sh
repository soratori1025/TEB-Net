#!/bin/bash
# ============================================================================
# Bruce Benchmark — TEB-Net v14.2 Evaluation
# 
# Reproduces the evaluation protocol from:
#   Bruce et al., "Machine learning predictions of thermochemical properties
#   for aliphatic carbon and oxygen species", Fuel 384, 133999 (2025)
#
# Usage:
#   ./run_benchmark.sh              # Full pipeline (data prep + train + eval)
#   ./run_benchmark.sh --prep-only  # Only prepare data
#   ./run_benchmark.sh --eval-only  # Only evaluate (if training is done)
#   ./run_benchmark.sh --quick      # Quick test with 1 seed only
# ============================================================================
set -e

PYTHON="/home/nvsinh1/brats_segmentation/teb-net-model/.venv/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Parse args
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

# ── Step 1: Prepare Dataset ──
if [ "$EVAL_ONLY" = false ]; then
    echo "[Step 1/3] Preparing WUDILY-CHO dataset..."
    $PYTHON prepare_dataset.py
    echo ""
fi

# ── Step 2: Train all seeds ──
if [ "$PREP_ONLY" = false ] && [ "$EVAL_ONLY" = false ]; then
    echo "[Step 2/3] Training TEB-Net v14.2 (5 seeds)..."
    
    if [ "$QUICK" = true ]; then
        # Quick mode: single seed
        echo "  (Quick mode: training seed 42 only)"
        $PYTHON train_tebnet.py \
            --seed 42 --split_seed 42 \
            --max_epochs $MAX_EPOCHS --patience $PATIENCE \
            --gpu 0
    else
        # Full mode: all 5 seeds in parallel
        $PYTHON run_all_seeds.py \
            --max_epochs $MAX_EPOCHS \
            --patience $PATIENCE \
            --gpus "$GPUS"
    fi
    echo ""
fi

# ── Step 3: Evaluate & Generate Report ──
if [ "$PREP_ONLY" = false ]; then
    echo "[Step 3/3] Generating evaluation report..."
    $PYTHON evaluate.py
    echo ""
fi

echo "============================================================"
echo " Benchmark complete!"
echo " Report: results/benchmark_report.md"
echo "============================================================"
