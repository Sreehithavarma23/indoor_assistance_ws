#!/usr/bin/env bash
# Reproducible NLU v2 build → train/compare → slot eval → frozen e2e.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=

echo "== Build dataset =="
python -m src.data_pipeline --seed 42 --max-per-family 3

echo "== Compare intent models =="
python -m src.nlu.compare_models --seed 42

echo "== Slot evaluation =="
python -m src.nlu.evaluate_slots

echo "== Frozen e2e (text) =="
python -m src.pipeline.evaluate_frozen

echo "== Dry-run demo =="
python -m src.simulation.dry_run "Take me to the kitchen." --json | head -n 40

echo "Done. See data/nlu_v2/DATASET_REPORT.md and results/nlu_v2/"
