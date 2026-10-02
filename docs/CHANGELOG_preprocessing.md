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
