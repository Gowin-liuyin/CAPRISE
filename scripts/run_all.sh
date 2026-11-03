#!/usr/bin/env bash
# Full reproduction pipeline: module gates (in order) + data + experiments.
# Fails fast: the gate runner stops at the first failing module (user requirement).
#
# Usage: bash scripts/run_all.sh [--skip-data] [--quick]
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
export HF_HOME="$PWD/.cache/huggingface"
SKIP_DATA=0
for arg in "$@"; do
  case "$arg" in
    --skip-data) SKIP_DATA=1 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

echo "== 1/6 core protocol gates (M1-M9) =="
$PY -m gates.run_gates --from M1 --to M9

echo "== 2/6 data preparation (M0) =="
if [ "$SKIP_DATA" = "1" ]; then
  echo "skipped (--skip-data)"
else
  if [ ! -f data/meta.json ]; then
    env -u ALL_PROXY "$PY" scripts/prepare_data.py --m 50000
  else
    echo "data/meta.json exists, reuse"
  fi
  $PY -m gates.run_gates --from M0 --to M0
fi

echo "== 3/6 theory / query-DP gates (M12, M13) =="
$PY -m gates.run_gates --from M12 --to M13

echo "== 4/6 ppRAG pipeline gates (M14-M16) =="
$PY -m gates.run_gates --from M14 --to M16

echo "== 5/6 privacy + efficiency gates (M10, M11, M17, M18) =="
$PY -m gates.run_gates --from M10 --to M11
$PY -m gates.run_gates --from M17 --to M18

echo "== 6/6 experiments =="
$PY experiments/exp5_query_dp.py
$PY experiments/exp1_efficiency.py
$PY experiments/exp2_retrieval.py
$PY experiments/exp3_vec2text.py
$PY experiments/exp4_vector_analysis.py

echo
echo "ALL DONE. See results/ and REPRODUCTION_REPORT.md"
