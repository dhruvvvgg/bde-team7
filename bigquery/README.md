# BigQuery Groundwater Stress Pipeline Reimplementation

## 1. Executive Summary & Honesty Disclosures

> [!IMPORTANT]
> **Reimplementation Notice:** The original preprocessing pipeline was authored in Python (`pandas`, `numpy`, `scipy`) and serves as the definitive reference implementation. The work contained in this directory is a **clean reimplementation in Google BigQuery SQL and JavaScript UDFs**, designed to run all intensive data transformations directly inside Google Cloud BigQuery (`river-data-511316-j4.gw_stress`) rather than on local hardware. The BigQuery version must never be described as the original work.

> [!NOTE]
> **Upstream Raster Extraction:** BigQuery cannot natively decode or sample geospatial raster files (GeoTIFF). Therefore, the extraction of monthly precipitation from raw CHIRPS v3.0 rasters to well coordinates remains an upstream Python step (`scripts/04a_chirps.py`). The extracted table is uploaded to BigQuery as `gw_stress.chirps_monthly`. Every stage after this extraction point runs exclusively in BigQuery SQL.

> [!NOTE]
> **JavaScript User-Defined Functions (UDFs):** Where standard SQL lacks analytical primitives, exact JavaScript UDFs are used rather than silent approximations:
> 1. `theil_sen_slope(years ARRAY<INT64>, vals ARRAY<FLOAT64>)`: Used in `05_stage5_all_features.sql` to compute the exact Theil-Sen median pairwise slope over same-season readings in the preceding 5 years.
> 2. `student_t_adj_anomaly(gwl FLOAT64, mean FLOAT64, std FLOAT64, n INT64)`: Used in `05_stage5_all_features.sql` to compute small-sample-adjusted anomaly scores via Lentz continued fractions for the incomplete beta function ($F_t(t; n-1)$) and Algorithm AS241 for the inverse standard normal distribution ($\Phi^{-1}(p)$).

---

## 2. Pipeline Architecture & Execution Order

All scripts are located in `bigquery/sql/`. They use `CREATE OR REPLACE TABLE ... AS SELECT` (CTAS) to comply with BigQuery Sandbox constraints (no billing, no DML `INSERT`/`UPDATE`/`DELETE`).

| Step | SQL Script | Target Table | Description |
|---|---|---|---|
| **01** | `bigquery/sql/01_stage2_reshape.sql` | `gw_stress.stage2_long` | Reshapes wide CGWB observations (`Jan-00`..`Nov-22`, 92 rounds) to long format (253,828 rows) and generates surrogate `well_id` (`W` + 10 SHA-1 hex characters). |
| **02** | `bigquery/sql/02_stage3_clean.sql` | `gw_stress.stage3_clean` | Computes QC flags (`flag_zero`, `flag_negative`, `flag_deeper_than_well`, modified z-score `flag_extreme`) and standardizes text strings. |
| **03** | `bigquery/sql/03_stage4_rainfall.sql` | `gw_stress.stage4_rain` | Computes antecedent 1m, 3m, 6m rainfall ending strictly at month $M-1$. Returns `NULL` if any antecedent month is missing. |
| **04** | `bigquery/sql/04_stage5_baselines.sql` | `gw_stress.stage5_baselines` | Calculates expanding seasonal baselines ($K=5$, 0.10 m standard deviation floor), anomaly z-scores, rolling-10 baselines, and completeness metrics. |
| **05** | `bigquery/sql/05_stage5_all_features.sql` | `gw_stress.stage5_all_features` | Computes Theil-Sen 5-year trends, Student-t adjusted anomalies, historical lags, water year fluctuations, and NE monsoon metrics. |
| **06** | `bigquery/sql/06_stage6_stress.sql` | `gw_stress.stage6_stress` | Implements parameterized SQL function `classify_stress` for primary, rolling-10, and t-adjusted categories; derives composite `suspect_reading` flags. |
| **07** | `bigquery/sql/07_stage7_final_table.sql` | `gw_stress.final_features` | Assembles the final 84-column contract table with strict type casting and `CLUSTER BY state, well_id`. |
| **08** | `bigquery/sql/08_splits.sql` | `gw_stress.splits` | Generates benchmark split definitions (`eval_eligible`, `split_chrono`, `state_fold`, `no_may_state`) using round coverage $\ge 70\%$. |

### Leakage Verification Test
- `bigquery/tests/test_leakage_perturbation.sql`: Clones `stage2_long` and `chirps_monthly`, injects massive anomalies (+99.99 m and +9,999 mm) for all dates on or after 2015, re-runs feature extraction, and asserts with `ASSERT NOT EXISTS` that zero historical features before 2015 were altered.

---

## 3. Reproduction Instructions

### Prerequisites
1. Google Cloud SDK (`gcloud` and `bq` CLI) authenticated to GCP.
2. Target project set: `river-data-511316-j4`.
3. Input tables uploaded in dataset `gw_stress`:
   - `gw_stress.raw_gwl_wide` (from `dataset/groundwater_data/Output/CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv`)
   - `gw_stress.chirps_monthly` (from `dataset/chirps/chirps_monthly_by_well.parquet`)
   - `gw_stress.well_crosswalk` (from `dataset/well_crosswalk.csv`)
   - `gw_stress.ref_final_table` (from `processed/gw_features_by_state/`)
   - `gw_stress.ref_splits` (from `processed/splits.parquet`)

### Running the SQL Pipeline
Execute each script in sequence:
```bash
bq query --nouse_legacy_sql < bigquery/sql/01_stage2_reshape.sql
bq query --nouse_legacy_sql < bigquery/sql/02_stage3_clean.sql
bq query --nouse_legacy_sql < bigquery/sql/03_stage4_rainfall.sql
bq query --nouse_legacy_sql < bigquery/sql/04_stage5_baselines.sql
bq query --nouse_legacy_sql < bigquery/sql/05_stage5_all_features.sql
bq query --nouse_legacy_sql < bigquery/sql/06_stage6_stress.sql
bq query --nouse_legacy_sql < bigquery/sql/07_stage7_final_table.sql
bq query --nouse_legacy_sql < bigquery/sql/08_splits.sql
```

### Running Tests and Parity Validation
```bash
# 1. Leakage test
bq query --nouse_legacy_sql < bigquery/tests/test_leakage_perturbation.sql

# 2. Parity against reference tables
python bigquery/run_parity.py
python bigquery/run_parity_splits.py
```

---

## 4. Sandbox Expiry and Offline Parquet Exports

All tables in the BigQuery Sandbox expire 60 days after creation.
- **Table Creation Date:** October 12, 2026
- **Sandbox Expiration Date:** December 11, 2026

To ensure permanent reproducibility and reviewability without relying on the cloud dataset lifetime, the completed tables have been exported locally into `bigquery/exports/`:
- `bigquery/exports/splits.parquet` (0.17 MB, 253,828 rows × 12 columns)
- `bigquery/exports/final_features.parquet` (~18 MB, 253,828 rows × 84 columns)

Both files adhere strictly to repository size limits (< 50 MB each).
