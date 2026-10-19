#!/bin/bash
source /home/nvsinh1/brats_segmentation/teb-net-model/.venv/bin/activate
export PYTHONPATH=$(pwd)
python -u run_pipeline.py > pipeline_resume.log 2>&1
