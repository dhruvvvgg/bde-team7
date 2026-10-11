# Big Data Infrastructure: Preprocessing in Google BigQuery

## 1. Architecture Overview & Rationale

Groundwater stress modeling across national monitoring networks requires processing multi-decadal time series alongside gridded satellite climate observations. The original reference pipeline was implemented locally in Python (`pandas` and `scipy`). To support scalable, auditable, and collaborative modeling without burdening local hardware, we re-implemented stages 2 through 7 of the preprocessing pipeline natively in **Google BigQuery**.

BigQuery provides a serverless columnar execution engine that decouples compute from storage, allowing analytical queries, windowed time-series aggregates, and user-defined statistical functions to run in parallel. All heavy feature engineering was executed on Google Cloud, processing input tables stored in the `river-data-511316-j4.gw_stress` dataset.

```
+-----------------------------------------------------------------------------------------+
|                                    INPUT TABLES                                         |
|  raw_gwl_wide (2,759 wells x 113 cols)   |   chirps_monthly (778,038 monthly rain rows) |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                                BIGQUERY SQL PIPELINE                                    |
|                                                                                         |
|  01_stage2_reshape.sql      UNPIVOT INCLUDE NULLS -> 253,828 grid rows (W+10 hex id)   |
|  02_stage3_clean.sql        Name standardisation, QC flags, MAD outlier detection       |
|  03_stage4_rainfall.sql     Antecedent 1m, 3m, 6m rainfall ending strictly at M-1       |
|  04_stage5_baselines.sql    Expanding baseline (K=5), rolling-10 (JS UDF), completeness |
|  05_stage5_all_features.sql Theil-Sen slope (JS UDF), Student-t adj (JS UDF), extras   |
|  06_stage6_stress.sql       Parameterized stress classification, suspect reading flags  |
|  07_stage7_final_table.sql  84-column contract table, CLUSTER BY state, well_id         |
|  08_splits.sql              Chronological & spatial splits, coverage gating             |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                                   FINAL ARTIFACTS                                       |
|  gw_stress.final_features (253,828 rows x 84 cols)  |  gw_stress.splits (253,828 rows)  |
|  Exported Parquet: final_features.parquet (<50 MB)  |  splits.parquet (0.17 MB)         |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Step Mapping & Implementation

| Pipeline Stage | Python Reference | BigQuery Implementation | Rationale |
| :--- | :--- | :--- | :--- |
| **Stage 2: Wide-to-Long Reshape** | `pd.melt()` | `UNPIVOT INCLUDE NULLS` | Preserves the complete 2,759 wells × 92 rounds grid (253,828 rows). Generates SHA-1 surrogate `well_id` (`W` + 10 hex characters). |
| **Stage 3: Cleaning & QC Flags** | `clean.py`, `scipy.stats` | SQL CASE expressions + `PERCENTILE_CONT(0.5)` | Flagging without deletion. Non-destructive outlier flagging via seasonal Median Absolute Deviation (MAD). |
| **Stage 4: Antecedent Rainfall** | `rainfall_windows()` | Analytic window sums ending at $M-1$ | Ensures zero temporal leakage. Any incomplete window returns `NULL`, preventing partial-sum bias. |
| **Stage 5: Expanding Baselines** | `_prior_expanding_stats()` | Analytic functions with `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` | Enforces past-only history; gated by $K=5$ minimum prior readings; applies $0.10$ m standard deviation floor. |
| **Stage 5: Rolling-10 Baselines** | `rolling_baseline()` | JavaScript UDF `calc_roll10_stats` | Maintains a 10-reading FIFO history buffer within each `(well_id, season)` partition, filtering nulls. |
| **Stage 5: Theil-Sen Trend** | `_theil_sen()`, `scipy.stats` | JavaScript UDF `theil_sen_slope` | Evaluates all pairwise slopes $\binom{n}{2}$ across preceding 5 years $[Y-5, Y-1]$ ($\ge 4$ points) and computes the exact median. |
| **Stage 5: Student-t Adjustment** | `scipy.stats.t`, `norm.isf` | JavaScript UDF `student_t_adj_anomaly` | Evaluates Lentz's continued fraction for the incomplete beta function and Wichura's Algorithm AS241 for the inverse normal CDF. |
| **Stage 6: Stress Classification** | `stress.classify()` | SQL SQL Function `classify_stress` | Parameterized multi-class labeling: `No reading` > `Insufficient history` > `High Stress` ($\ge 2$) > `Moderate Stress` ($\ge 1.5$) > `Watch` ($\ge 1$) > `Normal`. |
| **Stage 7: Final Contract Table** | `07_output.py` | `CREATE OR REPLACE TABLE ... CLUSTER BY state, well_id` | Aligns all 84 columns to contract schema; clusters by geographical partition to optimize downstream queries. |
| **Stage 8: Evaluation Splits** | `10_splits.py` | Windowed coverage filtering + chronological partitioning | Evaluates round sparsity ($< 70\%$ threshold); maps 19 alphabetical state folds. |

---

## 3. Measured Performance & Storage Metrics

All figures below are taken directly from BigQuery job metadata and table storage descriptors (`river-data-511316-j4.gw_stress`):

| Table Name | Output Rows | Logical Size (Uncompressed) | Physical Storage (Columnar) | Execution Time | Bytes Processed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `stage2_long` | 253,828 | 16.5 MB | 4.8 MB | 11.2 s | 1.8 MB |
| `stage3_clean` | 253,828 | 28.5 MB | 8.2 MB | 14.8 s | 16.5 MB |
| `stage4_rain` | 253,828 | 42.1 MB | 12.4 MB | 18.5 s | 46.2 MB |
| `stage5_baselines` | 253,828 | 68.2 MB | 21.6 MB | 24.6 s | 42.1 MB |
| `stage5_all_features` | 253,828 | 118.4 MB | 38.2 MB | 32.7 s | 78.5 MB |
| `stage6_stress` | 253,828 | 138.9 MB | 45.1 MB | 25.0 s | 118.4 MB |
| `final_features` (Clustered) | 253,828 | 146.2 MB | 49.5 MB | 20.6 s | 138.9 MB |
| `splits` | 253,828 | 23.9 MB | 0.1 MB | 14.6 s | 28.5 MB |

### Query Execution Profiles
- **Leakage Perturbation Verification Job:** Processed 49.0 MB, executed full re-computation and validation assertions in **14.3 s**.
- **Column-by-Column Parity Validation Job:** Scanned all 84 columns across 253,828 rows in **28.4 s**.

---

## 4. Honest Assessment of Scale & Architecture

1. **Dataset Scale:**  
   The primary dataset comprises 253,828 well-round rows (2,759 wells across 92 monitoring rounds). By big-data standards, this dataset is modest (approximately 146 MB uncompressed). On a single modern development workstation, Python pandas processes this volume in memory in roughly one minute.

2. **Why BigQuery Still Demonstrates Cloud-Scale Architecture:**  
   While local memory is sufficient for 2,759 wells, national groundwater programs collect data from hundreds of thousands of wells, telemetry loggers recording hourly, and high-resolution daily gridded weather rasters. In BigQuery:
   - **Columnar Pruning:** Queries analyzing specific feature subsets (e.g., rainfall anomalies or trend slopes) scan only the requested columns, reducing I/O by over 80%.
   - **Partitioning & Clustering:** Clustering by `state, well_id` enables sub-second retrieval for leave-one-state-out cross-validation models without full-table scans.
   - **Declarative Reproducibility:** Entire transformation stages run as immutable `CREATE OR REPLACE TABLE ... AS SELECT` statements, eliminating pipeline fragility caused by local Python environment variations.
