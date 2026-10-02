# Seasonal Groundwater Stress Assessment and Cross-Region Generalization Across India

Big Data Essentials, team 7. This repository holds the data preprocessing pipeline.
Its output feeds the modeling, evaluation and dashboard stages.

- **Final table:** `processed/gw_features_by_state/`. This is Parquet partitioned by
  `state_slug`, with 253,828 rows: 2,759 wells × 92 rounds from 2000 to 2022.
- **Data quality:** `reports/data_quality.md`.
- **Column dictionary and modeling caveats:** `docs/handoff_for_modeling.md`.

## Data sources

| Source | Where | Notes |
|---|---|---|
| CGWB groundwater levels with specific yield (Figshare, [10.6084/m9.figshare.29293877.v3](https://doi.org/10.6084/m9.figshare.29293877.v3), CC BY 4.0) | `dataset/groundwater_data/` (committed, never modified) | We use `Output/CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv`, the authors' refined set: 2,759 wells, 19 states, 365 districts. |
| CHIRPS v3.0 monthly rainfall (Climate Hazards Center) | `dataset/chirps/chirps_monthly_by_well.parquet` (committed, 4.4 MB) | This holds values sampled at each well for 1999-07 to 2022-12. The raw rasters are never committed. |

## Reproducing the pipeline from scratch

You need Python 3.11 or later. Stage 4a also needs HTTPS access to `data.chc.ucsb.edu`, unless the
committed CHIRPS table is already present. FTP is not used.

```bash
pip install -r requirements.txt

python scripts/01_inspect.py        # stage 1: profile the dataset -> reports/stage1_inspection.md
python scripts/02_reshape.py        # stage 2: wide -> long, well_id crosswalk -> dataset/well_crosswalk.csv
python scripts/03_clean.py          # stage 3: placeholders, duplicates, names, flags -> reports/stage3_cleaning.md
python scripts/04a_chirps.py        # stage 4a: CHIRPS download -> sample -> delete (resumable, ~15 min)
python scripts/04b_rain_windows.py  # stage 4b: 1/3/6-month antecedent rainfall -> reports/stage4_rainfall.md
python scripts/05_features.py       # stage 5: baseline, anomaly, trend, rain deviation -> reports/stage5_features.md
python scripts/06_stress.py         # stage 6: stress categories and threshold sensitivity -> reports/stage6_stress.md
python scripts/07_output.py         # stage 7: partitioned Parquet + reports/data_quality.md

python -m pytest -q                 # tests, including the leakage tests
```

How the stages fit together:

- **Intermediate files.** Stages 2 to 6 write to `processed/interim/`, which is gitignored and can be regenerated.
- **CHIRPS download.** Stage 4a skips any month already in the committed CHIRPS table. With the table
  complete, it does nothing and needs no network. Delete the table to force a fresh download.
- **Rasters.** These are downloaded one at a time, sequentially, into `dataset/chirps/raw/`, which is
  gitignored. Each is deleted right after it has been sampled.
- **Determinism.** Every step is deterministic, with no random sampling. `well_id` is a hash of the
  well's state, district, station name and coordinates, so it is identical on every run. The original
  `Station Code` was corrupted upstream by scientific-notation rounding.
  `dataset/well_crosswalk.csv` maps `well_id` back to it.

### Key decisions

All of these are documented in code docstrings and in `reports/`.

- **Wells.** All 2,759 wells are kept.
- **Outliers.** None are removed again. The authors already applied 3σ filtering, so this pipeline only flags extremes.
- **Zero readings.** The 11 readings of exactly 0.0 m are kept and flagged.
- **Rainfall windows.** These end at month M−1 for a reading in month M. An assertion and a pytest test enforce this.
- **Seasonal baseline.** This uses earlier same-season readings only, and needs at least K = 5 of them.
  K = 3 and K = 8 are reported as sensitivity checks.
- **Stress thresholds.** The defaults are 1, 1.5 and 2 SD on the positive (deeper-than-usual) anomaly.
  `No reading` takes precedence over `Insufficient history`.
- **Partitioning.** The output is partitioned by state, because the project's main test is cross-region generalization.
