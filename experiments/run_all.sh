#!/usr/bin/env bash
# Full experiment: policy comparison for three label delays x five seeds, then the detector benchmark.
set -euo pipefail
PY=${PY:-.venv/bin/python}
for delay in 1 3 6; do
  for seed in 0 1 2 3 4; do
    $PY -m experiments.run_policy_comparison --label-delay "$delay" --seed "$seed"
  done
done
$PY -m experiments.run_policy_comparison --label-delay 3 --seed 0 --skip-shadow --mlflow-uri sqlite:///results/mlflow.db \
  --out-dir results/policy_comparison_mlflow
$PY -m experiments.run_detector_benchmark --seeds 10
