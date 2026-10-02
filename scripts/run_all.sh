#!/usr/bin/env bash
# Regenerate every output from the committed inputs. Never downloads anything:
# the CHIRPS tables in dataset/chirps/ are committed, and the download scripts
# (04a, 04c, 09_chirps_extension) are deliberately NOT called here.
#
# Usage: bash scripts/run_all.sh        (from anywhere)
set -euo pipefail
cd "$(dirname "$0")/.."

for f in dataset/chirps/chirps_monthly_by_well.parquet \
         dataset/chirps/chirps_monthly_by_well_3x3.parquet \
         dataset/chirps/chirps_monthly_by_well_2023_2024.parquet; do
  [ -f "$f" ] || { echo "missing committed input $f (run the download scripts first)"; exit 1; }
done

run() { echo "==> $*"; python "$@" > /dev/null; }
run scripts/01_inspect.py              # stage 1 report
run scripts/02_reshape.py              # long table + crosswalk
run scripts/03_clean.py                # cleaning + flags
run scripts/04b_rain_windows.py        # rainfall windows (from committed CHIRPS)
run scripts/05_features.py             # features (incl. Parts A, A2, B, B2)
run scripts/06_stress.py               # stress labels + suspect flags
run scripts/07_output.py               # partitioned final table + data_quality.md
run scripts/04d_chirps_sampling_check.py   # Part C report (reads committed tables)
run scripts/08_validity.py             # Part D report
run scripts/09a_extension_match.py     # Part E matching + coverage
run scripts/09b_extension_features.py  # Part E features (+ leakage proof)
run scripts/10_splits.py               # Part F splits
if [ -f scripts/11_handoff_dictionary.py ]; then run scripts/11_handoff_dictionary.py; fi
run scripts/12_figures.py              # review figures, tables, captions
run scripts/13_architecture.py         # architecture diagram
echo "==> pytest"; python -m pytest -q
