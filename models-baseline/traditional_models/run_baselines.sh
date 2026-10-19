#!/usr/bin/env bash
# Table-1 baselines under the DeepTherm protocol: 5 random splits + the complexity split.
# A run whose results.json already exists is skipped, so the script can be re-started safely.
#   bash run_baselines.sh            # everything
#   ONLY=ml  bash run_baselines.sh   # conventional ML only
#   ONLY=gnn bash run_baselines.sh   # GNNs only
set -u

source ../../.venv/bin/activate
export PYTHONPATH=/home/nvsinh1/brats_segmentation/teb-net-model/models-s3/tebnet-v1:/home/nvsinh1/brats_segmentation/teb-net-model

OUT=${OUT:-benchmark_results_baselines}
GPU=${GPU:-0}
ONLY=${ONLY:-all}
SPLITS=("--split_type random --split_seed 0" "--split_type random --split_seed 1"
        "--split_type random --split_seed 2" "--split_type random --split_seed 3"
        "--split_type random --split_seed 4" "--split_type complexity --split_seed 0")

run() { echo; echo ">>> $*"; "$@" || echo "!!! failed: $*"; }

for S in "${SPLITS[@]}"; do
  if [[ $ONLY == all || $ONLY == ml ]]; then
    run python baselines_ml.py --csv ../deeptherm_data.csv --model ridge --features prior          $S --out_dir $OUT
    run python baselines_ml.py --csv ../deeptherm_data.csv --model rf    --features ecfp           $S --out_dir $OUT
    run python baselines_ml.py --csv ../deeptherm_data.csv --model rf    --features physics        $S --out_dir $OUT
    run python baselines_ml.py --csv ../deeptherm_data.csv --model xgb   --features physics        $S --out_dir $OUT
    run python baselines_ml.py --csv ../deeptherm_data.csv --model xgb   --features physics+ecfp   $S --out_dir $OUT
  fi
  if [[ $ONLY == all || $ONLY == gnn ]]; then
    for A in gin gcn gat attentivefp; do
      CUDA_VISIBLE_DEVICES=$GPU run python baselines_gnn.py --csv ../deeptherm_data.csv --arch $A --targets all $S --out_dir $OUT
    done
  fi
done

echo
echo "Evaluate (same command as for TEB-Net; add --exclude_file excluded_rows.txt for the cleaned test):"
echo "  python deeptherm_eval.py --runs $OUT --csv baselines_summary.csv"