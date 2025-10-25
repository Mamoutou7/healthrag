#!/usr/bin/env bash
python3 - <<'PY'
from pathlib import Path
from src.eval.evaluator import HealthEvaluator
e = HealthEvaluator()
e.run_and_save(Path("data/gold/eval_set.json"), Path("models/eval_report.json"))
PY
