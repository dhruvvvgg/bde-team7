# BigQuery Preprocessing Implementation Plan

## 1. Honesty & Scope Declarations

> [!IMPORTANT]
> **Reimplementation Notice:** The original preprocessing pipeline was authored in Python (`pandas` / `numpy` / `scipy`) and serves as the ground-truth **reference implementation**. The work described here is a **reimplementation in Google BigQuery SQL and JavaScript UDFs**, designed to execute all heavy computations directly inside the data warehouse and validated column-by-column against the reference Python outputs. The BigQuery pipeline must never be described as the original work.

> [!NOTE]
> **Upstream Raster Processing:** BigQuery cannot natively read or sample GeoTIFF rasters. Therefore, extracting monthly rainfall from raw CHIRPS v3.0 rasters into a well-month table remains an upstream Python step (`scripts/04a_chirps.py`). The resulting extraction table (`dataset/chirps/chirps_monthly_by_well.parquet`) has been loaded into BigQuery as `gw_stress.chirps_monthly`. All antecedent window aggregations, normals, deviations, and z-scores downstream of that table are computed purely in BigQuery SQL.

> [!NOTE]
> **JavaScript User-Defined Functions (UDFs):** Where SQL lacks native mathematical primitives, JavaScript UDFs are employed without silent approximations:
> 1. `theil_sen_slope(years_array, vals_array)`: Exact Theil-Sen median pairwise slope calculation over same-season readings in preceding 5 years.
> 2. `student_t_adj_anomaly(gwl, mean, std, n)`: Computes small-sample-adjusted normal-equivalent anomaly scores via exact Student-t CDF and inverse standard normal quantile functions ($\Phi^{-1}(F_t(t; n-1))$).

---

## 2. Input Tables Verification in BigQuery

The dataset `gw_stress` was verified in project `river-data-511316-j4`. All five reference and input tables have been inspected with their exact BigQuery row counts and schemas:

| Table Name | BigQuery Row Count | Columns | Schema Highlights | Source File in Repository |
|---|---:|---:|---|---|
| `raw_gwl_wide` | 2,759 | 113 | 20 metadata cols, 92 observation cols (`Jan-00`..`Nov-22`), `Reference_Sy` | `dataset/groundwater_data/Output/CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv` |
| `chirps_monthly` | 778,038 | 4 | `well_id:STRING`, `year:INT64`, `month:INT64`, `precip_mm:FLOAT64` | `dataset/chirps/chirps_monthly_by_well.parquet` |
| `well_crosswalk` | 2,759 | 8 | `well_id`, `state`, `district`, `station_name`, `lat`, `lon`, corrupted codes | `dataset/well_crosswalk.csv` |
| `ref_final_table` | 253,828 | 84 | All 84 target feature and label columns | `processed/gw_features_by_state/` (2,759 wells × 92 rounds) |
| `ref_splits` | 275,900 | 12 | `well_id`, `period_label`, `date`, `eval_eligible`, `split_chrono`, etc. | `processed/splits.parquet` (main + extension) |

---

## 3. Python Stages to BigQuery SQL Mapping

| Stage | Python Source Module | BigQuery Script & Target Table | Implementation Mechanism | Hard-to-Reproduce Nuances & Handling |
|---|---|---|---|---|
| **Stage 2: Reshape** | `src/gwstress/reshape.py`<br>`scripts/02_reshape.py` | `01_stage2_reshape.sql`<br>$\rightarrow$ `gw_stress.stage2_long` | BigQuery `UNPIVOT` on 92 reading columns, joining with `well_crosswalk` for surrogate `well_id` | **Surrogate Key:** Computed via SHA-1 over `(state, district, station_name, lat, lon)`. Verified that BigQuery SQL formula `CONCAT('W', SUBSTR(TO_HEX(SHA1(...)), 1, 10))` matches `well_crosswalk` exactly. We join to `well_crosswalk` to ensure 100% key fidelity while asserting hash equivalence. |
| **Stage 3: Cleaning & QC** | `src/gwstress/clean.py`<br>`scripts/03_clean.py` | `02_stage3_clean.sql`<br>$\rightarrow$ `gw_stress.stage3_clean` | SQL conditional expressions, `REGEXP_REPLACE` name standardization, window functions for whole-record modified z-score | **Modified z-score:** Uses whole-record seasonal median and MAD per well. In SQL, median is computed via `PERCENTILE_CONT(gwl, 0.5) OVER (PARTITION BY well_id, season)`. Values $\le -900$ converted to NULL. |
| **Stage 4: Antecedent Rainfall** | `src/gwstress/rainfall.py`<br>`scripts/04b_rain_windows.py` | `03_stage4_rainfall.sql`<br>$\rightarrow$ `gw_stress.stage4_rain` | Window functions / self-joins on `chirps_monthly` using continuous month index `year*12 + month - 1` | **Strict Leakage Guard:** Windows end strictly at $M-1$ ($M$ is never used). If any month in $M-w..M-1$ is missing, sum returns `NULL` (never partial sum). Embedded `ASSERT` checks max source month $< M$. |
| **Stage 5: Expanding & Rolling Baselines** | `src/gwstress/features.py`<br>`scripts/05_features.py` | `04_stage5_baselines.sql`<br>$\rightarrow$ `gw_stress.stage5_baselines` | Window aggregates with `BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` partitioned by `(well_id, season)` | **Minimum History $K=5$:** If earlier non-missing readings $< 5$, baseline and anomaly are `NULL`. Standard deviation uses `STDDEV_SAMP` (ddof=1) and is floored at 0.10 m. Rolling 10 uses non-null history array / sliding window. |
| **Stage 5: Theil-Sen & Small-Sample Adjustment** | `src/gwstress/features.py`<br>`scripts/05_features.py` | `05_stage5_theil_sen_tadj.sql`<br>$\rightarrow$ `gw_stress.stage5_advanced` | JavaScript UDF for Theil-Sen slope over preceding 5 years; JavaScript UDF for Student-t adjustment | **Theil-Sen Ties & Even Counts:** Median of pairwise slopes must follow `np.median` exact average-of-middle behavior. **Student-t Adjustment:** Accurate rational approximations for $F_t(t; n-1)$ and $\Phi^{-1}(p)$ to avoid floating-point overflow/underflow at large $t$. |
| **Stage 5: Extra Features & Calendar** | `src/gwstress/features_extra.py` | `06_stage5_extra_features.sql`<br>$\rightarrow$ `gw_stress.stage5_extra` | SQL window lags (`LAG(..., 1)`, `LAG(..., 4)`), annual pivot for May-to-Nov fluctuation, monsoon & NE-monsoon window sums | **Hydrological Year & NE Monsoon:** Oct–Dec rainfall of $Y-1$ (`last_ne_year`). May-to-Nov recharge `gw(Nov, Y-1) - gw(May, Y-1)`. Monsoon YTD rainfall from 1 June. Expanding normal over prior monsoons only. |
| **Stage 6: Stress Classification & Validity** | `src/gwstress/stress.py`<br>`src/gwstress/validity.py`<br>`scripts/06_stress.py` | `07_stage6_stress.sql`<br>$\rightarrow$ `gw_stress.stage6_stress` | SQL Function / parameterized `CASE` statements for stress categories and composite suspect flags | **Precedence Order:** `No reading` $\rightarrow$ `Insufficient history` $\rightarrow$ `High Stress` ($\ge 2.0$) $\rightarrow$ `Moderate Stress` ($\ge 1.5$) $\rightarrow$ `Watch` ($\ge 1.0$) $\rightarrow$ `Normal`. Composite `suspect_reading` combining the 3 suspicious zeros, depth violations, and extreme anomalies. |
| **Stage 7: Final Feature Table** | `scripts/07_output.py`<br>`src/gwstress/contract.py` | `08_stage7_final_table.sql`<br>$\rightarrow$ `gw_stress.final_features` | Column projection and type casting to match contract, table clustering by `state, well_id` | **Schema Parity:** Exactly 84 columns, identical names, types, order, and null policies. Clustered by `state, well_id` for optimal query filtering. |
| **Part F: Evaluation Splits** | `src/gwstress/splits.py`<br>`scripts/10_splits.py` | `09_splits.sql`<br>$\rightarrow$ `gw_stress.splits` | SQL join with `splits.json` logic: round coverage calculation, chronological assignment, state folds | **Eligibility & Exclusions:** `eval_eligible` requires non-missing reading, $n \ge K$, and round coverage $\ge 70\%$. Identifies the 7 sparse rounds in 2000–2022. Assigns `state_fold` (0..18) and `no_may_state`. |

---

## 4. Planned 84 Output Columns Mapping

All 84 columns defined in `src/gwstress/contract.py` will be produced:

1. **Keys & Calendar (9 cols):** `well_id`, `date`, `period_label`, `year`, `month`, `season`, `water_year`, `water_year_index`, `season_order`.
2. **Current Target Reading (1 col):** `gwl_m_bgl`.
3. **Static Covariates (10 cols):** `state`, `state_slug`, `district`, `station_name`, `lat`, `lon`, `well_type`, `aquifer_type`, `well_depth_m`, `specific_yield`.
4. **Stage-3 QC Flags (6 cols):** `flag_zero`, `flag_extreme`, `flag_negative`, `flag_deeper_than_well`, `suspect_reading`, `suspect_reason`.
5. **Antecedent & Historical Rainfall (26 cols):** `rain_1m_mm`, `rain_3m_mm`, `rain_6m_mm`, `rain_1m_normal_mm`, `rain_3m_normal_mm`, `rain_6m_normal_mm`, `rain_1m_dev_mm`, `rain_3m_dev_mm`, `rain_6m_dev_mm`, `rain_1m_dev_pct`, `rain_3m_dev_pct`, `rain_6m_dev_pct`, `rain_3m_z`, `rain_3m_z_capped`, `rain_6m_z`, `rain_6m_z_capped`, `rain_12m_mm`, `rain_monsoon_ytd_mm`, `last_monsoon_year`, `rain_last_monsoon_mm`, `rain_last_monsoon_dev_mm`, `rain_last_monsoon_dev_pct`, `last_ne_year`, `rain_last_ne_mm`, `rain_last_ne_dev_mm`, `rain_last_ne_z_capped`.
6. **Expanding Seasonal Baseline & Anomalies (8 cols):** `n_prior_same_season`, `baseline_mean_m`, `baseline_std_m`, `flag_std_floored`, `anomaly_z`, `anomaly_z_capped`, `anomaly_tadj`, `anomaly_tadj_capped`.
7. **Rolling-10 Baseline & Anomalies (7 cols):** `n_roll10`, `baseline_roll10_mean_m`, `baseline_roll10_std_m`, `flag_roll10_std_floored`, `anomaly_roll10`, `anomaly_roll10_capped`, `stress_roll10`.
8. **Trend & Completeness (4 cols):** `trend_5y_m_per_yr`, `completeness_to_date`, `well_completeness_full_period`, `well_has_no_series_for_season`.
9. **Groundwater History Lags & Fluctuations (11 cols):** `gw_prev_round_m`, `gw_last_obs_m`, `rounds_since_last_obs`, `gw_same_season_lag1y_m`, `gw_same_season_lag2y_m`, `gw_prev_change_m`, `gw_delta_from_prev_round_m`, `gw_max_to_date_m`, `gw_min_to_date_m`, `n_readings_to_date`, `gw_fluct_prev_wy_m`.
10. **Stress Classifications (2 cols):** `stress_category` (primary label), `stress_tadj`.

---

## 5. JavaScript UDF Specifications

### UDF 1: `theil_sen_slope`
- **Location:** Embedded in `05_stage5_theil_sen_tadj.sql`.
- **Signature:** `theil_sen_slope(years ARRAY<INT64>, vals ARRAY<FLOAT64>) RETURNS FLOAT64`
- **Mathematical Logic:**
  - For each pair $(i, j)$ with $i < j$, compute slope $s_{ij} = (vals[j] - vals[i]) / (years[j] - years[i])$.
  - Sort all pairwise slopes ascending.
  - Return median: if count is odd, middle element; if even, average of the two middle elements.

### UDF 2: `student_t_adj_anomaly`
- **Location:** Embedded in `05_stage5_theil_sen_tadj.sql`.
- **Signature:** `student_t_adj_anomaly(x FLOAT64, mean FLOAT64, s FLOAT64, n INT64) RETURNS FLOAT64`
- **Mathematical Logic:**
  - Standardized score $t = (x - mean) / (s_{eff} \cdot \sqrt{1 + 1/n})$ where $s_{eff} = \max(s, 0.10)$, degrees of freedom $\nu = n - 1$.
  - Regularized incomplete beta function $I_x(a, b)$ used to compute $F_t(t; \nu)$.
  - Rational approximation (Acklam / Beasley-Springer-Moro algorithm) to compute standard normal quantile $\Phi^{-1}(p)$.
  - Upper and lower tails handled symmetrically to prevent rounding to $\pm \infty$. Values clamped to $[-38.0, 38.0]$ before final $\pm 5.0$ capping.

---

## 6. Leakage Prevention and Verification Strategy

1. **Window Frames Ending at 1 PRECEDING:** All seasonal expanding baselines, rainfall normals, and rolling windows enforce `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` within properly ordered date partitions.
2. **Rainfall Antecedent Bound:** All rainfall lookbacks compute antecedent totals for months $\le M-1$. SQL queries include explicit `ASSERT` statements checking that `MAX(src_month) < obs_month`.
3. **Perturb-the-Future SQL Test:**
   - In `bigquery/tests/test_leakage_perturbation.sql`, create a temporary copy of `stage2_long` and `chirps_monthly`.
   - Modify every reading and rainfall value strictly on or after cutoff date (e.g. November 2006) with large perturbations ($+100$ m, $+5000$ mm).
   - Re-run all feature logic on the perturbed copy.
   - Execute an `ASSERT NOT EXISTS` query confirming that zero feature values on or before August 2006 differ from the unperturbed table.

---

## 7. BigQuery Clustering and Storage Choices

- **Clustering:** Final table `gw_stress.final_features` will be clustered by `(state, well_id)`.
  - **Rationale:** The project's primary benchmark is cross-region evaluation (leave-one-state-out). In BigQuery, clustering by `state` co-locates rows for each of the 19 states into contiguous storage blocks, dramatically reducing bytes scanned during state-based model training and regional evaluation. Secondary clustering by `well_id` accelerates per-well longitudinal queries.
  - **No Date Partitioning:** The dataset spans 2000–2022 (92 rounds total, 253,828 rows, ~146 MB). Partitioning by date would create 92 tiny partitions (< 2 MB each), incurring metadata overhead without query pruning benefits.

---

## 8. Order of Execution & Stop Point

1. **Review and Approval:** Review this plan.
2. **Execution (Phase 1):** Upon approval, execute scripts `01_stage2_reshape.sql` through `09_splits.sql` in sequence via `bq query`.
3. **Parity Validation:** Execute the comparison query joining `gw_stress.final_features` with `gw_stress.ref_final_table` and `gw_stress.splits` with `gw_stress.ref_splits` to verify row counts, column types, categorical matches, and float differences ($\le 10^{-9}$).
4. **Parity Report:** Generate `bigquery/PARITY_REPORT.md` and report results.
