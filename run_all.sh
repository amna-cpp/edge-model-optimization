#!/usr/bin/env bash
# run_all.sh - reproducible end-to-end pipeline for Assignment 03 (Q1 + Q2).
# Every step logs to logs/<step>.log (and the console). Seeds are fixed (seed=42).
#
#   bash run_all.sh            # full pipeline (~1.5-2 h on an Apple M-series laptop)
#   bash run_all.sh --quick    # tiny smoke-test run (a few minutes) to check the environment
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
mkdir -p logs checkpoints results onnx_models

if [[ "${1:-}" == "--quick" ]]; then
  B="--epochs 1";  Q="--epochs 1 --max-images 2000"; U="--iterations 2 --ft-epochs-per-iter 1 --final-ft-epochs 1"; S="--ft-epochs 1"
else
  B="";            Q="";                              U="";                                                          S=""
fi

export PYTHONWARNINGS=ignore          # hide torch.ao deprecation chatter in the logs
step() { echo; echo "######## $1 ########"; shift; "$@" 2>&1 | tee "logs/$STEP.log"; }

STEP=01_train_baseline;     step "Baseline FP32 training"          $PY train_baseline.py $B
STEP=02_ptq;                step "Post-training quantisation"      $PY ptq.py
STEP=03_qat;                step "Quantisation-aware training"     $PY qat.py $Q
STEP=04_unstructured;       step "Unstructured pruning"            $PY prune_unstructured.py $U
STEP=05_structured;         step "Structured channel pruning"      $PY prune_structured.py $S
STEP=06_inference_demo;     step "Inference demo (5 models)"       $PY inference.py --samples 8
STEP=07_benchmark;          step "Benchmark (5 models)"            $PY benchmark.py
STEP=08_export_onnx;        step "ONNX export"                     $PY export_onnx.py
STEP=09_onnx_benchmark;     step "ONNX CPU inference benchmark"    $PY onnx_inference.py
STEP=10_report_figures;     step "Report figures"                  $PY make_report_assets.py

echo; echo "Pipeline finished. Results: results/*.csv   Checkpoints: checkpoints/   ONNX: onnx_models/"
