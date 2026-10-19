#!/usr/bin/env bash
# Build a download-ready archive of the TEB-Net research tree.
#
# Excludes, with reasons:
#   .venv/                    6.8G  rebuildable: python -m venv + pip install -e .
#   .pytest_cache/            tool cache
#   **/__pycache__/           byte code
#   models-baseline/sota/*/.git  ~2.3G of third-party history, all four repos are
#                                public on GitHub (chemprop, HC-Thermochemistry,
#                                DeepTherm); re-clone instead of shipping it
#   framework/.git/           30M, pushed to origin (0 commits ahead)
#   *.pyc, .DS_Store, *.log in lightning_logs (huge, regenerable)
#
# Everything else is kept, including every checkpoint, results.json,
# predictions.npz, the manuscript tables in check/, and figures/.
set -euo pipefail

SRC=/home/nvsinh1/brats_segmentation/teb-net-model
STAMP=$(date +%Y-%m-%d)
OUT=/home/nvsinh1/tebnet_research_${STAMP}.zip

cd "$(dirname "$SRC")"

# -y stores symlinks as symlinks; there are none left under the research tree
# except inside third-party repos, which is what we want.
zip -r -q "$OUT" "$(basename "$SRC")" \
  -x "$(basename "$SRC")/.venv/*" \
  -x "$(basename "$SRC")/.pytest_cache/*" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x "*/models-baseline/sota/*/.git/*" \
  -x "$(basename "$SRC")/framework/.git/*" \
  -x "$(basename "$SRC")/framework/.pytest_cache/*" \
  -x "*/.DS_Store"

echo "wrote $OUT"
ls -lh "$OUT"
unzip -l "$OUT" | tail -1
