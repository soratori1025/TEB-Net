#!/usr/bin/env bash
set -euo pipefail
SRC=/home/nvsinh1/brats_segmentation/teb-net-model
STAMP=$(date +%Y-%m-%d)
OUT=/home/nvsinh1/tebnet_research_${STAMP}.zip
cd "$(dirname "$SRC")"
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
