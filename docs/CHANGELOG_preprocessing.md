# Preprocessing changelog

**Rollback point (pre-improvement baseline): `dab7477368c299f222fb5e4c4efb4a90243ec9fe`.**
This commit holds the accepted output of stages 1–7. To restore that state, check the commit out
(`git checkout dab7477368c299f222fb5e4c4efb4a90243ec9fe -- .`) or diff against it.

Every change to the committed outputs after that commit is listed below. Each entry gives what
changed, why, and a before/after summary. Existing column definitions are not changed without
explicit approval; any proposed change to an existing column is listed under "Proposed, awaiting
approval" first.

## Proposed, awaiting approval

(none)

## Changes

### Part A: extra leakage-safe features (added columns only)

- **What:** 24 new columns were merged into `processed/gw_features_by_state/`. The code is in
  `src/gwstress/features_extra.py`, called from `scripts/05_features.py`.
  - **Groundwater history:** `gw_prev_round_m`, `gw_last_obs_m`, `rounds_since_last_obs`,
    `gw_same_season_lag1y_m`, `gw_same_season_lag2y_m`, `gw_prev_change_m`,
    `gw_delta_from_prev_round_m`, `gw_max_to_date_m`, `gw_min_to_date_m`, `n_readings_to_date`,
    `gw_fluct_prev_wy_m`. Only `gw_delta_from_prev_round_m` uses the current reading; it is a target
    candidate, not an input.
  - **Rainfall:** `rain_12m_mm`, `last_monsoon_year`, `rain_last_monsoon_mm`,
    `rain_last_monsoon_dev_mm`, `rain_last_monsoon_dev_pct`, `rain_3m_z`, `rain_6m_z`,
    `rain_3m_z_capped`, `rain_6m_z_capped`, `rain_monsoon_ytd_mm`.
  - **Calendar:** `water_year`, `water_year_index`, `season_order`. `season` already existed.
- **Why:** these give the modeling stage lagged groundwater state, recharge history and longer
  rainfall memory, all computed past-only.
- **Before/after:**
  - Rows: 253,828 → 253,828, unchanged.
  - Columns: 44 → 68.
  - Existing columns: **0 values changed in all 44**. This was checked cell by cell against the
    rollback commit, with float tolerance and missing values compared as equal.
  - Size: the folder grew from 29.9 MB to 47 MB, and the largest file from 5.5 MB to 9.0 MB.
- **Design choices worth knowing:**
  - "Change from previous round" as an input is `gw_prev_change_m` = reading(t−1) − reading(t−2).
    The literal current-minus-previous version is `gw_delta_from_prev_round_m`, which is class C.
  - `rain_12m_mm` is NaN for the Jan-2000 and May-2000 rounds, because CHIRPS here starts in
    1999-07. `rain_monsoon_ytd_mm` is NaN for the same reason (Jun-1999 is missing), and
    `rain_last_monsoon_mm` is NaN for the Jan, May and Aug 2000 rounds.
  - The rainfall z-scores are heavy-tailed (|z| > 5 in 0.6–0.8% of rows, mostly dry pre-monsoon
    windows), so capped ±5 versions were added for models.
- **Tests:**
  - The perturb-the-future test now rebuilds rainfall windows from monthly CHIRPS. It runs at 4
    cutoffs (Jan, May, Aug and Nov 2006), perturbing both readings and monthly rain, and covers
    every column except the two documented whole-record columns.
  - A meta-test proves the check fails on a deliberately leaky column.
  - Exact-value tests cover each new feature (`tests/test_features_extra.py`).
  - 37 tests pass.

### Part B: stationary (rolling 10-reading) stress variant (added columns only)

- **What:** 7 new columns. The code is `features.rolling_baseline()`, and `scripts/06_stress.py`
  adds the label.
  - Baseline: `n_roll10`, `baseline_roll10_mean_m`, `baseline_roll10_std_m`,
    `flag_roll10_std_floored`.
  - Anomaly: `anomaly_roll10`, `anomaly_roll10_capped` (±5).
  - Label: `stress_roll10`, from the same `classify()` with the same thresholds and labels.
  - The baseline uses the last 10 non-missing earlier readings of the same well and season, needs at
    least 5, and applies the same 0.10 m std floor.
  - A comparison section was added to `reports/data_quality.md` (section 9).
- **Why:** to test whether the expanding baseline's lag explains the excess High Stress rate.
- **Before/after:**
  - Rows: 253,828, unchanged.
  - Columns: 68 → 75.
  - Existing columns: **0 values changed**. `stress_category` is still the primary label.
    Checked with `scripts/compare_to_baseline.py`.
- **Finding:** the rolling variant does **not** lower the High Stress rate.

  | | Expanding | Rolling |
  |---|---:|---:|
  | High Stress, share of 166,306 readings classified under both | 6.38% | 6.90% |

  The two labels differ on 10,900 readings (6.6%), mostly between adjacent categories. The main
  cause of the excess is small-sample std estimation: with 5–10 earlier readings, even
  independent Gaussian data would give about 4.7–6.3% at z ≥ 2. Drought years add the rest.
- **Tests:**
  - Exact-window test (last 10 earlier readings, NaN skipped, current reading excluded).
  - Level-shift test showing the rolling mean forgets an old level.
  - The perturb-the-future test covers the new columns automatically.
  - 40 tests pass.

### Part C: rainfall sampling check (new file only; existing outputs untouched)

- **What:**
  - New file `dataset/chirps/chirps_monthly_by_well_3x3.parquet` (8.1 MB): 778,038 rows with
    `precip_nearest_mm`, `precip_3x3_mm` and `n_valid_3x3`.
  - CHIRPS was re-extracted from 1999-07 to 2022-12 over HTTPS, one file at a time, with each raster
    deleted after use.
  - The analysis is in `reports/chirps_sampling_check.md`, and the flagged wells are listed in
    `reports/chirps_sampling_flagged_wells.csv`.
  - New code: `chirps.sample_nearest_and_3x3()`, `chirps.build_table_3x3()`,
    `scripts/04c_chirps_3x3.py` and `scripts/04d_chirps_sampling_check.py`. A synthetic-raster test
    covers the edge, nodata and negative cases.
- **Why:** to check whether the single nearest 0.05° cell is representative of each well's area.
- **Before/after:**
  - `chirps_monthly_by_well.parquet` and the feature table: **unchanged**.
  - The re-extracted nearest values match the committed table on all 778,038 well-months (max diff
    0.0000 mm).
- **Result:**

  | Measure | Value |
  |---|---|
  | Monthly correlation, nearest vs 3×3 | r = 0.9991 |
  | Mean absolute difference | 3.43 mm/month (2.7%) |
  | Annual correlation | r = 0.9989 |
  | Wells with nodata in their 3×3 block | 0 |
  | Wells flagged for large differences | 13, all in steep terrain |

  Recommendation: **do not switch** (see the report).
- **Bug found during development:** the first version of the 3×3 sampler counted the edge row twice
  at the raster boundary. It was caught by a synthetic test and fixed before the download. No
  committed output used the buggy version.

### Part D: data-validity checks (added columns + new report)

- **What:**
  - New columns in the feature table:
    - `suspect_reading` (bool): descriptive, class B.
    - `suspect_reason` (`zero_suspicious` / `deeper_than_well` / `extreme_anomaly`, joined with
      `;` when a reading has two).
  - Code: `src/gwstress/validity.py`, called from `scripts/06_stress.py`. The three suspicious zeros
    are listed in `config.SUSPICIOUS_ZEROS`.
  - New report `reports/data_validity.md` (`scripts/08_validity.py`). It covers coordinates,
    co-location, CHIRPS nodata, suspect readings by reason and state, and rainfall plausibility
    per state, with a hand-reviewed interpretation.
- **Why:** to give one documented QC flag for sensitivity analyses, and to confirm the inputs are
  physically plausible.
- **Before/after:**
  - Rows: 253,828, unchanged.
  - Columns: 75 → 77.
  - Existing columns: **0 values changed**, checked with `scripts/compare_to_baseline.py`.
  - No reading was deleted or modified.
- **Results:**
  - **Coordinates:** 0 wells outside the bounding box. 0 wells with a nodata CHIRPS cell, whether
    nearest or anywhere in the 3×3 block. 0 exactly co-located wells; 1 pair about 100 m apart
    (Chanderi, MP).
  - **Suspect readings:** 2,135 (0.97%) in 1,244 wells. By reason: 1,612 extreme anomaly, 537 deeper
    than the well, 3 suspicious zeros; 17 readings have two reasons.
  - **Rainfall:** no implausible state totals. The flagged wells are explained by Tamil Nadu's rain
    shadow (low Jun–Sep rainfall) and the Udupi Ghats crest (over 7,000 mm a year).

### Part B2: small-sample-adjusted anomaly (added columns only)

- **What:**
  - New columns: `anomaly_tadj`, `anomaly_tadj_capped` (±5) and `stress_tadj` (same `classify()`,
    thresholds and labels).
  - Formula: t = (x − mean) / (s_eff · √(1 + 1/n)), then z_adj = Φ⁻¹(F_t(t; n − 1)).
  - It uses the same expanding baseline, K = 5 and 0.10 m floor as `anomaly_z`.
  - New report section 10 in `reports/data_quality.md`.
  - The existing `anomaly_z` also uses ddof = 1. The difference is that the new score accounts for
    the uncertainty in the estimated mean and uses t-distribution tails, as the docstring explains.
- **Why:** to remove the small-sample inflation of tail rates identified in Part B.
- **Before/after:**
  - Rows: 253,828, unchanged.
  - Columns: 77 → 80.
  - Existing columns: 0 values changed. `stress_category` is still the primary label.
- **Result:**

  | | `stress_category` | `stress_tadj` | Nominal |
  |---|---:|---:|---:|
  | High Stress | 6.38% | 4.18% | 2.28% |

  - 54% of the excess High Stress disappears.
  - 10,611 classified readings (6.4%) change category, all towards Normal.
  - The remaining excess clusters in 2015–2018 and 2009.
- **Tests:**
  - Large-n agreement with `anomaly_z` (within 0.03).
  - Monte Carlo on i.i.d. normal data at n = 5 and 8 (40,000 draws each): the raw score's tail rate
    is more than 1.8× nominal, while the adjusted rate is within 0.4 points of 2.28%.
  - NaN below K, finite values otherwise, and correct caps.
  - The perturb-the-future test explicitly asserts the new columns are covered.
  - 45 tests pass.

### Part A2: north-east monsoon (Oct–Dec) rainfall features (added columns only)

- **What:**
  - New columns: `last_ne_year`, `rain_last_ne_mm`, `rain_last_ne_dev_mm` and
    `rain_last_ne_z_capped` (±5).
  - Every round in year Y uses Oct, Nov and Dec of year Y−1. That is the last fully completed
    Oct–Dec season, because Dec of year Y falls after every round in Y.
  - The normal and z-score use earlier Oct–Dec seasons only, at least 5 of them.
  - The same `_month_sum()` leakage assertion applies.
  - New report section 11 in `reports/data_quality.md`, comparing the Jun–Sep and Oct–Dec rainfall
    share by state.
- **Why:** Jun–Sep features miss the main rainy season in Tamil Nadu and an important part of it in
  Andhra Pradesh.
- **Before/after:**
  - Rows: 253,828, unchanged.
  - Columns: 80 → 84.
  - Existing columns: 0 values changed.
- **Result:**
  - Tamil Nadu: Oct–Dec is 41.4% of annual rain, against 40.1% for Jun–Sep.
  - Andhra Pradesh: 27.1% Oct–Dec.
  - Kerala: 14.3% Oct–Dec.
  - All other states: under 12%.
- **Tests:**
  - Exact months for every season, including Nov Y not using Oct Y.
  - The normal uses only earlier years.
  - The perturb-the-future test explicitly asserts the new columns are covered.
  - 47 tests pass.

### Part D follow-up: documentation only

- **What:** added three sections to `docs/handoff_for_modeling.md`:
  - "Using `suspect_reading`": do not drop flagged rows by default; run an exclusion sensitivity
    check instead; how likely each reason code is to be a true error.
  - "Region-specific notes": Tamil Nadu monsoon timing and the Chanderi pair of nearby wells.
  - "Rainfall sensitivity resource": the 3×3 CHIRPS table and how to join it.
- **Before/after:** no data changes.

### Part E, step 1: extension matching proof and coverage (new report only)

- **What:**
  - New code: `src/gwstress/extension.py` and `scripts/09a_extension_match.py`.
  - New report: `reports/extension_2023_2024.md`, sections 1–3. It was written before any extension
    features were built.
- **Matching:**
  - All 2,759 wells were matched by surrogate `well_id`, with 0 unmatched and 0 raw-coordinate
    mismatches.
  - One well (`W0388be4a8d`, Raidih) has two stage-4 rows. It was resolved by taking the row whose
    2000–2022 readings equal the main table exactly. The other row is a sparse twin whose two
    2023–24 values are identical to the kept row's.
  - All 253,828 main cells for 2000–2022 are identical to the matched stage-4 rows.
- **Coverage:**

  | Period | Share of cells with a reading |
  |---|---:|
  | 2018–2022 | 80.0% |
  | 2023–2024 | 30.9% |

  - Coverage falls from 69.8% (Jan-23) to 8.0% (Nov-24).
  - 255 wells have no 2023–24 reading at all.
  - Delhi has 0%. Tamil Nadu and Jharkhand stay above 80%.
- **Zero readings:** 87 in 2023–24 (68 of them in May-23), against 11 in 2000–2022. They look like
  placeholders. They are kept and flagged.
- **Before/after:** main table unchanged.

### Part E, step 2: extension features (new separate outputs)

- **What:**
  - `dataset/chirps/chirps_monthly_by_well_2023_2024.parquet`: nearest cell, 2023-01 to 2024-12,
    66,216 rows, 0 NaN. HTTPS only, sequential, rasters deleted.
  - `processed/extension_2023_2024/extension_features.parquet`: 22,072 rows × 84 columns (the main
    table's columns plus `is_extension`), 5.6 MB.
  - Code: `scripts/09_chirps_extension.py` and `scripts/09b_extension_features.py`.
  - `chirps.build_table()` gained optional `first`/`last` month arguments. The defaults are
    unchanged, so the main table's build is identical.
- **How:**
  - The same functions run on the combined 2000–2024 series and the 1999-07 to 2024-12 rainfall
    record.
  - The 87 extension zero readings are flagged `suspect_reason = zero_extension`.
  - The whole-record descriptive columns (`flag_extreme`, `well_completeness_full_period`,
    `well_has_no_series_for_season`) use 2000–2024 for the extension rows.
- **Leakage proof:** recomputing the 2000–2022 rows with 2023–24 data present reproduces the
  committed main table exactly. That covers 253,828 rows across all 80 columns apart from the three
  whole-record ones. The script asserts this on every run.
- **Before/after:**
  - Main table: unchanged (0 rows, 0 values).
  - Main rainfall table: unchanged.

### Part F: shared splits file (new file)

- **What:**
  - `processed/splits.parquet`: 275,900 rows (253,828 main + 22,072 extension), keyed by
    (`well_id`, `period_label`).
  - Columns: `eval_eligible`, `exclusion_reason`, `split_chrono`, `state_fold`, `no_may_state`,
    `source`, `date`, `year`, `season` and `state`.
  - Cutoffs live in `config/splits.json`. The code is `src/gwstress/splits.py` and
    `scripts/10_splits.py`, and the summary is `reports/splits_summary.md`.
- **Rules:**
  - `eval_eligible` = reading exists, at least 5 earlier same-season readings, and the round is not
    sparse.
  - A round is sparse when coverage is under 70% (main) or under 15% (extension). The separate
    extension threshold exists because the 70% rule would exclude all 8 extension rounds; Jan-23 is
    at 69.8%.
  - Excluded main rounds: May-20, May-21, Aug-12, Jan-16, Aug-20, May-16, Nov-18.
  - Excluded extension rounds: May-24, Aug-24, Nov-24.
- **Eval-eligible rows by split:**

  | Split | Years | Eligible rows |
  |---|---|---:|
  | train | 2000–2015 | 104,437 |
  | val | 2016–2017 | 15,513 |
  | test | 2018–2022 | 40,630 |
  | ext | 2023–2024 | 5,989 |

- **Before/after:** feature tables unchanged. This is a separate file, as specified.
