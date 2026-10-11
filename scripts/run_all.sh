#!/usr/bin/env bash
# Spark is the default preprocessing engine. The historical pandas pipeline
# remains available explicitly for independent parity checks and old figures.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${GW_ENGINE:-spark}" = "pandas" ]; then
  exec bash scripts/run_pandas_reference.sh
fi
python scripts/run_spark.py "$@"
python -m pytest -q tests/test_spark_pipeline.py
