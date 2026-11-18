#!/usr/bin/env bash
# Reproduce every phase-2 result from the raw data (about 1.5-2 hours on a 10-core laptop).
# Run from the repository root with the virtual environment active and the dataset in data/raw/.
set -euo pipefail
export PYTHONWARNINGS=ignore
mkdir -p outputs/logs
run() { echo "== $1"; shift; "$@" 2>&1 | tee "outputs/logs/$(date +%H%M%S)_step.log" | tail -5; }
run "tests"                 python -m pytest -q
run "phase-1 EDA"           python -m src.data_exploration.run_eda
run "5 models + CV"         python -m src.models.train_compare
run "model selection"       python -m src.models.select_model
MODEL=$(python -c "import json;print(json.load(open('results/models/selected_model.json'))['selected'])")
run "radius study ($MODEL)" python -m src.features.radius_study --model "$MODEL"
run "Bayesian optimisation" python -m src.optimization.bayes_opt --model "$MODEL" --trials 25
run "HMM experiments"       python -m src.hmm.run_hmm --model "$MODEL" --probs tuned
run "export artefacts"      python scripts/build_final_artifacts.py
run "validate artefacts"    python scripts/validate_artifacts.py --trip-dir data/raw/student_test --trip 00000052
run "experiment figures"    python -m src.visualization.experiment_plots
run "prediction maps"       python -m src.visualization.prediction_maps --partition student_test --trips 00000052 00000040 00000077
run "all student-test trips through the API service" python scripts/run_api_rehearsal.py
run "technical report"      python scripts/build_report.py
