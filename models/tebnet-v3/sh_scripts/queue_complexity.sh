#!/bin/bash
echo "Waiting for run_ureel_cv.sh to finish..."
while pgrep -f "run_ureel_cv.sh" > /dev/null; do
    sleep 30
done
echo "Ureel CV finished. Starting Complexity Split..."
bash sh_scripts/run_complexity_ensemble.sh
