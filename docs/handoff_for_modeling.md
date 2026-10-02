# Handoff for modeling

This is the entry point for anyone using the preprocessing outputs. Do not train on anything without
reading "Known data limitations" and "How to use the splits" first.

## What to load

```python
import pandas as pd

main = pd.read_parquet("processed/gw_features_by_state")              # 253,828 rows x 84 columns
kl = pd.read_parquet("processed/gw_features_by_state",
                     filters=[("state_slug", "=", "kerala")])         # one state
ext = pd.read_parquet("processed/extension_2023_2024/extension_features.parquet")  # 22,072 rows, holdout only
splits = pd.read_parquet("processed/splits.parquet")                   # 275,900 rows (main + extension)

df = main.merge(splits, on=["well_id", "period_label"], suffixes=("", "_split"))   # 1:1, tested
```

**What one row is:** one well in one CGWB monitoring round (Jan, May, Aug or Nov). The main table
covers 2,759 wells × 92 rounds from 2000 to 2022, and 219,258 rows have a reading. The extension holds
the same wells for the 8 rounds of 2023–2024. It is a separate holdout that must never be mixed into
training data.

**Keys:** (`well_id`, `date`) or, equivalently, (`well_id`, `period_label`). Both are unique.

**Primary label:** `stress_category`, decided and fixed. `stress_tadj` and `stress_roll10` are
sensitivity labels.

**Primary rainfall:** the nearest CHIRPS cell, decided and fixed. A 3×3 table exists for sensitivity
checks only (see the end of this document).

## Column dictionary

Each column has one of three classes:

- **A**: safe model input. It uses only earlier rounds, rainfall up to month M−1, or static
  attributes.
- **B**: descriptive only. It is an identifier, a QC flag, or a whole-record statistic that uses
  future information. **Never a model input.**
- **C**: label or target candidate. It uses the current reading, so use it as an input only when
  lagged from an earlier round.

The "NaN when" column says when a value can be missing. "never" means the contract test enforces no
missing values.

<!-- BEGIN GENERATED COLUMN DICTIONARY (scripts/11_handoff_dictionary.py from src/gwstress/contract.py; do not edit by hand) -->
84 columns in the main table (+ `is_extension` in the extension table). Class counts: A: 60, B: 13, C: 11.


**Keys and calendar**

| column | class | type | NaN when | range | description |
|---|:-:|---|---|---|---|
| `well_id` | B | string | never |  | Surrogate well key (hash of state, district, station name, lat, lon). Key, never a feature; `dataset/well_crosswalk.csv` maps it to the original codes. |
| `date` | B | datetime | never |  | First day of the monitoring month (the day is not published). Part of the key. |
| `period_label` | B | string | never |  | Original round label, e.g. `May-07`. Part of the key in `processed/splits.parquet`. |
| `year` | A | int | never | 2000 to 2024 | Calendar year. |
| `month` | A | int | never | 1 to 11 | Round month: 1, 5, 8 or 11. |
| `season` | A | category | never |  | `post_monsoon_rabi` (Jan), `pre_monsoon` (May), `monsoon` (Aug), `post_monsoon_kharif` (Nov). |
| `water_year` | A | int | never | 1999 to 2024 | Start year of the Jun-May hydrological year that contains the round. |
| `water_year_index` | A | int | never | 0 to 25 | `water_year - 1999`. |
| `season_order` | A | int | never | 0 to 3 | Position within the water year: Aug=0, Nov=1, Jan=2, May=3. |

**Reading (target)**

| column | class | type | NaN when | range | description |
|---|:-:|---|---|---|---|
| `gwl_m_bgl` | C | float | no reading in this round | 0 to 200 | Depth to groundwater, m below ground level (larger = deeper = more stressed). Target for level forecasting. |

**Static covariates**

| column | class | type | NaN when | range | description |
|---|:-:|---|---|---|---|
| `state` | A | string | never |  | Standardised state name. Use for splits; as a feature it weakens the cross-region test (treat as B for held-out-state evaluation). |
| `state_slug` | B | category | never |  | Partition key (file-system-safe state). |
| `district` | A | string | never |  | Standardised district name (same caveat as `state`). |
| `station_name` | B | string | never |  | Station name (identifier). |
| `lat` | A | float | never | 6 to 37.5 | Latitude (WGS84). |
| `lon` | A | float | never | 68 to 97.5 | Longitude (WGS84). |
| `well_type` | A | string | never |  | Dug / bore / tube / dug-cum-bore well. |
| `aquifer_type` | A | string | not recorded by CGWB (127 wells) |  | Unconfined / semi-confined / confined. |
| `well_depth_m` | A | float | never | 0 to 400 | Recorded well depth (m). May be outdated (see `suspect_reading`). |
| `specific_yield` | A | float | never | 0 to 0.2 | Authors' `Reference_Sy`: 5 aquifer-class values from the hydrogeological map. Treat as ordinal or categorical. |

**Groundwater history (Part A)**

| column | class | type | NaN when | range | description |
|---|:-:|---|---|---|---|
| `gw_prev_round_m` | A | float | previous round missing | 0 to 200 | Reading of the immediately previous round. |
| `gw_last_obs_m` | A | float | no earlier reading | 0 to 200 | Most recent earlier non-missing reading. |
| `rounds_since_last_obs` | A | float | no earlier reading | 1 to 100 | How many rounds back `gw_last_obs_m` is. |
| `gw_same_season_lag1y_m` | A | float | that round missing | 0 to 200 | Same season, previous year. |
| `gw_same_season_lag2y_m` | A | float | that round missing | 0 to 200 | Same season, two years earlier. |
| `gw_prev_change_m` | A | float | t-1 or t-2 missing | -∞ to ∞ | reading(t-1) - reading(t-2). |
| `gw_delta_from_prev_round_m` | C | float | t or t-1 missing | -∞ to ∞ | reading(t) - reading(t-1). Uses the current reading: target candidate, NOT an input. |
| `gw_max_to_date_m` | A | float | no earlier reading | 0 to 200 | Deepest earlier reading. |
| `gw_min_to_date_m` | A | float | no earlier reading | 0 to 200 | Shallowest earlier reading. |
| `n_readings_to_date` | A | int | never | 0 to 120 | Number of earlier non-missing readings. |
| `gw_fluct_prev_wy_m` | A | float | May or Nov of Y-1 missing | -∞ to ∞ | gw(Nov, Y-1) - gw(May, Y-1): last completed monsoon's depth change (negative = recharge). |

**Rainfall (stage 4, Parts A, A2)**

| column | class | type | NaN when | range | description |
|---|:-:|---|---|---|---|
| `rain_1m_mm` | A | float | never | 0 to 6000 | CHIRPS rainfall in month M-1 (nearest 0.05 deg cell). |
| `rain_3m_mm` | A | float | never | 0 to 12000 | Rainfall over months M-3..M-1. |
| `rain_6m_mm` | A | float | never | 0 to 20000 | Rainfall over months M-6..M-1. |
| `rain_1m_normal_mm` | A | float | < 5 earlier years (2000-2004) | 0 to 6000 | Mean of `rain_1m_mm` for the same well and season over earlier years only. |
| `rain_3m_normal_mm` | A | float | < 5 earlier years (2000-2004) | 0 to 12000 | Same, 3-month window. |
| `rain_6m_normal_mm` | A | float | < 5 earlier years (2000-2004) | 0 to 20000 | Same, 6-month window. |
| `rain_1m_dev_mm` | A | float | no normal | -∞ to ∞ | `rain_1m_mm` minus its earlier-years normal. |
| `rain_3m_dev_mm` | A | float | no normal | -∞ to ∞ | Same, 3-month window. |
| `rain_6m_dev_mm` | A | float | no normal | -∞ to ∞ | Same, 6-month window. |
| `rain_1m_dev_pct` | A | float | no normal, or normal < 5 mm | -100 to ∞ | Percentage deviation from the normal. |
| `rain_3m_dev_pct` | A | float | no normal, or normal < 5 mm | -100 to ∞ | Same, 3-month window. |
| `rain_6m_dev_pct` | A | float | no normal, or normal < 5 mm | -100 to ∞ | Same, 6-month window. |
| `rain_3m_z` | A | float | < 5 earlier years, or earlier std < 1 mm | -∞ to ∞ | z of `rain_3m_mm` against the same well and season in earlier years. Heavy-tailed: prefer the capped version. |
| `rain_3m_z_capped` | A | float | as rain_3m_z | -5 to 5 | `rain_3m_z` clipped to +/-5 (use in models). |
| `rain_6m_z` | A | float | < 5 earlier years, or earlier std < 1 mm | -∞ to ∞ | Same, 6-month window. |
| `rain_6m_z_capped` | A | float | as rain_6m_z | -5 to 5 | `rain_6m_z` clipped to +/-5 (use in models). |
| `rain_12m_mm` | A | float | Jan/May 2000 (CHIRPS starts 1999-07) | 0 to 30000 | Rainfall over months M-12..M-1. |
| `rain_monsoon_ytd_mm` | A | float | Jan/May 2000 (June 1999 missing) | 0 to 30000 | Rainfall from 1 June of the current monsoon year up to M-1 (Jan: Jun-Dec; May: Jun-Apr; Aug: Jun-Jul; Nov: Jun-Oct). |
| `last_monsoon_year` | A | int | never | 1999 to 2024 | Year of the last completed Jun-Sep monsoon (Y for Nov rounds, Y-1 otherwise). |
| `rain_last_monsoon_mm` | A | float | Jan/May/Aug 2000 (June 1999 missing) | 0 to 20000 | Jun-Sep total of `last_monsoon_year`. |
| `rain_last_monsoon_dev_mm` | A | float | < 5 earlier monsoons | -∞ to ∞ | That total minus the well's mean over earlier monsoons. |
| `rain_last_monsoon_dev_pct` | A | float | < 5 earlier monsoons | -100 to ∞ | Same, in percent. |
| `last_ne_year` | A | int | never | 1999 to 2023 | Year of the last completed Oct-Dec (north-east) monsoon: always Y-1. |
| `rain_last_ne_mm` | A | float | never | 0 to 10000 | Oct+Nov+Dec rainfall of `last_ne_year`. |
| `rain_last_ne_dev_mm` | A | float | < 5 earlier Oct-Dec seasons | -∞ to ∞ | That total minus the well's mean over earlier Oct-Dec seasons. |
| `rain_last_ne_z_capped` | A | float | < 5 earlier seasons, or std < 1 mm | -5 to 5 | z of the Oct-Dec total against earlier seasons, clipped to +/-5. |

**Baselines, anomalies, labels (stage 5-6, Parts B, B2)**

| column | class | type | NaN when | range | description |
|---|:-:|---|---|---|---|
| `n_prior_same_season` | A | int | never | 0 to 30 | Number of earlier readings of the same well and season. |
| `baseline_mean_m` | A | float | < K=5 earlier same-season readings | 0 to 200 | Mean of all earlier same-season readings (expanding). |
| `baseline_std_m` | A | float | < K=5 earlier same-season readings | 0 to 200 | Std (ddof=1) of the same. May be ~0; the anomaly floors it at 0.10 m. |
| `flag_std_floored` | A | bool | never |  | `baseline_std_m` < 0.10 m and floored. |
| `anomaly_z` | C | float | no reading, or no baseline | -∞ to ∞ | (reading - baseline_mean) / max(std, 0.10). Raw, heavy-tailed. |
| `anomaly_z_capped` | C | float | as anomaly_z | -5 to 5 | `anomaly_z` clipped to +/-5 (use in models). |
| `anomaly_tadj` | C | float | as anomaly_z | -40 to 40 | Small-sample-adjusted normal-equivalent score: Phi^-1(F_t(t; n-1)), with t = (x - mean) / (s * sqrt(1 + 1/n)). |
| `anomaly_tadj_capped` | C | float | as anomaly_z | -5 to 5 | `anomaly_tadj` clipped to +/-5. |
| `stress_category` | C | category | never |  | PRIMARY label: No reading / Insufficient history / Normal / Watch / Moderate Stress / High Stress (thresholds 1 / 1.5 / 2 on `anomaly_z`). |
| `stress_tadj` | C | category | never |  | Same labels on `anomaly_tadj` (sensitivity label). |
| `n_roll10` | A | int | never | 0 to 10 | Number of earlier same-season readings in the rolling window (max 10). |
| `baseline_roll10_mean_m` | A | float | < 5 earlier same-season readings | 0 to 200 | Mean of the last 10 earlier same-season readings. |
| `baseline_roll10_std_m` | A | float | < 5 earlier same-season readings | 0 to 200 | Std (ddof=1) of the same. |
| `flag_roll10_std_floored` | A | bool | never |  | Rolling std < 0.10 m and floored. |
| `anomaly_roll10` | C | float | no reading, or no rolling baseline | -∞ to ∞ | Anomaly against the rolling baseline (raw). |
| `anomaly_roll10_capped` | C | float | as anomaly_roll10 | -5 to 5 | Clipped to +/-5. |
| `stress_roll10` | C | category | never |  | Same labels on `anomaly_roll10` (sensitivity label). |

**Trend and completeness**

| column | class | type | NaN when | range | description |
|---|:-:|---|---|---|---|
| `trend_5y_m_per_yr` | A | float | < 4 of the 5 earlier same-season readings | -∞ to ∞ | Theil-Sen slope over the same season, years Y-5..Y-1 (m/yr). |
| `completeness_to_date` | A | float | first round of each well | 0 to 1 | Share of the well's earlier rounds that have a reading. |
| `well_completeness_full_period` | B | float | never | 0 to 1 | Share of ALL rounds with a reading (uses future data). |
| `well_has_no_series_for_season` | B | bool | never |  | The well has no reading at all in this season over the whole record (uses future data; for grouping). |

**QC flags (stage 3, Part D)**

| column | class | type | NaN when | range | description |
|---|:-:|---|---|---|---|
| `flag_zero` | B | bool | never |  | Reading is exactly 0.0 m. |
| `flag_extreme` | B | bool | never |  | Modified z > 3.5 within the well-season over the WHOLE record (uses future data). QC only. |
| `flag_negative` | B | bool | never |  | Negative depth (none occur). |
| `flag_deeper_than_well` | B | bool | never |  | Reading deeper than the recorded well depth (usually an outdated depth record). |
| `suspect_reading` | B | bool | never |  | Composite QC flag (Part D). Do not drop by default; use for an exclusion sensitivity run. |
| `suspect_reason` | B | string | not suspect |  | `zero_suspicious`, `deeper_than_well`, `extreme_anomaly`, `zero_extension` (extension only), joined with `;`. |

**Extension only**

| column | class | type | NaN when | range | description |
|---|:-:|---|---|---|---|
| `is_extension` | B | bool | never |  | True for every row of the 2023-2024 extension table. |
<!-- END GENERATED COLUMN DICTIONARY -->

### Using a C column as an input

Lag it within the well. For example, take `anomaly_z_capped` from the previous round, or from the
same season last year, by grouping on `well_id` and shifting after sorting by `date`. The
`gw_*` history columns already do this for the reading itself.

### Which anomaly and rainfall z-scores to use

- Use the **capped** versions in models: `anomaly_z_capped`, `anomaly_tadj_capped`,
  `rain_3m_z_capped`, `rain_6m_z_capped` and `rain_last_ne_z_capped`. The raw z-scores have very
  heavy tails; for example `rain_6m_z` reaches 113.
- `anomaly_tadj` corrects for small-sample uncertainty in the baseline. With 5 to 10 earlier
  readings, the raw `anomaly_z` exceeds 2 far more often than a normal distribution would. See
  `reports/data_quality.md`, section 10.

## Known data limitations

| Issue | Size | Consequence |
|---|---|---|
| May 2020 and May 2021 almost empty | 0.9% of wells measured (COVID-19 period) | No labels in these rounds. May 2022 trends are mostly NaN (2,014 of 2,038 readings). |
| Aug 2012 almost empty | 2.5% of wells measured | Effectively no monsoon label for 2012. |
| Jan 2016 sparse | 19.1% of wells measured | Few labels, and not representative of all states. |
| Other rounds below 70% | Aug 2020 (58.5%), May 2016 (59.0%), Nov 2018 (66.7%) | Excluded from evaluation by `eval_eligible`. |
| No May series for 444 wells | All of Kerala (171) and West Bengal (68), 189 of 191 in Odisha, all 16 in Assam | No pre-monsoon labels for these states. `no_may_state` in the splits file marks them. |
| First K = 5 readings of each well-season have no baseline | 52,952 readings (24.2%), mostly 2000–2004 | Labels effectively start around 2005. |
| Small-sample tail inflation | High Stress is 6.38% (`stress_category`) vs 4.18% (`stress_tadj`) vs 2.28% nominal | Treat High Stress rates in early years as inflated. Report `stress_tadj` as a sensitivity check. |
| Jun–Sep features weak for Tamil Nadu | Oct–Dec is 41% of Tamil Nadu's rain | Use the `rain_last_ne_*` features (see the region notes below). |
| `Station Code` corrupted upstream | All wells | Use `well_id`. Joining to other CGWB data needs name and lat/lon matching. |
| Monitoring day unknown | All rows | Rainfall windows stop at M−1. Never add same-month rainfall. |
| **Extension 2023–24 not quality-filtered like the main data** | Coverage 30.9% (80.0% in 2018–22). Nov-24 has 8.0%; Delhi has no readings. 87 likely-placeholder zeros. | The extension readings did not pass the authors' repeated-value filter or the at-least-2-readings-per-year rule. Use the extension only as a stress test, exclude `zero_extension` rows, and report its results separately from the 2018–2022 test. |

## Extension holdout (2023–2024): rules of use

- **Secondary robustness check only.** The 2,759 wells were selected for completeness through 2022.
  They were **not** filtered for 2023–24 completeness, and the 2023–24 readings did not pass the
  authors' repeated-value filter.
- **Never pool** extension results with the 2018–2022 test set. Report them in a separate table.
- **Always report per-round reading counts** next to any extension metric (coverage falls from 69.8%
  in Jan-23 to 8.0% in Nov-24). `reports/extension_2023_2024.md` has the counts.
- Rounds with under 15% coverage (May-24, Aug-24, Nov-24) are not eval-eligible.
- **Zero readings:** the 87 `zero_extension` readings (68 in May-23) are likely placeholders. They
  are `eval_eligible = False` (`extension_zero_placeholder`) and are **excluded from every baseline,
  lag and history statistic** of later extension rounds, while their stored value (0.0) is unchanged.
  The 11 main-table zeros are treated as genuine readings: 8 are in monsoon rounds where a full well
  is plausible, and the other 3 are flagged as suspect only.

## How to use the splits

`processed/splits.parquet` is keyed by (`well_id`, `period_label`), and its cutoffs live in
`config/splits.json`. Edit that file and run `python scripts/10_splits.py` to regenerate it.

1. **Train and evaluate only on `eval_eligible == True`.** That means a reading exists, there are at
   least 5 earlier same-season readings, and the round is not sparse. `exclusion_reason` says why a
   row is out. The sparse rounds are listed in `reports/splits_summary.md`.
2. **Cross-region:** hold out whole states with `state_fold` (0–18). Never let a well appear in both
   train and test. The Chanderi pair (see below) is in one state, so state folds keep it together.
3. **Chronological:** `split_chrono` is `train` (2000–2015), `val` (2016–2017), `test` (2018–2022) or
   `ext` (2023–2024).

   | Split | Eval-eligible rows |
   |---|---:|
   | train | 104,437 |
   | val | 15,513 |
   | test | 40,630 |
   | ext | 5,989 |

   Rows that aren't eligible keep their period label, so always filter on `eval_eligible`.
4. **May evaluation:** report per-season metrics. For held-out states where `no_may_state` is true,
   pre-monsoon metrics cannot be computed. Say so rather than reporting zero.
5. **Class imbalance:** classified rows under `stress_category` are 81.5% Normal, 7.6% Watch, 4.5%
   Moderate Stress and 6.4% High Stress. Use macro-F1 and per-class recall.
6. **Sensitivity checks:** rerun headline results with `stress_tadj` and with flagged rows
   (`suspect_reading`) excluded.
7. **Never use class B columns as inputs.** All class A columns are strictly past-only per well. The
   perturb-the-future tests show this at 4 cutoffs, and the extension build shows that adding
   2023–24 data leaves every 2000–2022 row unchanged. They therefore do not leak across a
   chronological split.

## Using `suspect_reading`

`suspect_reading` / `suspect_reason` flag 2,135 readings (0.97%) in 1,244 wells. **Do not drop
flagged rows by default.** Instead, train and evaluate on the full data, then repeat the evaluation
with flagged rows excluded, and report both as a sensitivity check. How likely each reason is to be
a genuine error:

| Reason | Readings | How likely a true error | Notes |
|---|---:|---|---|
| `zero_suspicious` | 3 | **High** | These are isolated 0.0 m readings between much deeper neighbours (Ujjain Aug-19, Raisen Nov-19, Nellore Jan-18). |
| `extreme_anomaly` (\|anomaly_z\| > 5) | 1,612 | **Mixed** | Some are single bad readings and some are genuine sharp drops. Check `anomaly_tadj`: under the small-sample adjustment many of these are far less extreme. |
| `deeper_than_well` | 537 | **Low** | Mostly an outdated **well-depth record**, not a bad reading. The most-flagged well (Kanpur Dehat, `We3954794ef`) falls smoothly from 8 m to 19 m against a recorded depth of 14 m. A smooth series like that points to a deepened well, not 35 bad readings. |

## Region-specific notes

- **Tamil Nadu monsoon timing.** Oct–Dec brings 41% of Tamil Nadu's annual rain, against 40% for
  Jun–Sep. Andhra Pradesh gets 27% in Oct–Dec, and Kerala 14%. For these states, use the
  `rain_last_ne_*` features (Oct–Dec of the previous year) alongside the Jun–Sep ones. Expect weaker
  transfer to a held-out Tamil Nadu from models trained on Jun–Sep-dominated states. Eight wells in
  Erode, Tiruppur and Coimbatore get less than 150 mm in Jun–Sep; this is genuine rain shadow, not a
  data error.
- **Chanderi pair.** `Chanderi` (`Wc1a2cd0be0`) and `Chanderi(d)` (`W3a57256af7`) in Ashok Nagar,
  Madhya Pradesh, are about 100 m apart and share the same CHIRPS cell, so all their rainfall
  features are identical. They are kept as separate wells. Any split below state level must keep
  them in the same fold (for example `GroupKFold` on a location group).

## Rainfall sensitivity resource: 3×3 CHIRPS table

`dataset/chirps/chirps_monthly_by_well_3x3.parquet` (778,038 rows) has the following columns:

- `well_id`, `year`, `month`
- `precip_nearest_mm`: identical to the primary rainfall table
- `precip_3x3_mm`: mean of the 3×3 block of 0.05° cells around the well
- `n_valid_3x3`: number of valid cells in that block; 9 for every well-month

It is **not** used by any feature. The primary rainfall source is the nearest cell, by decision.
Use it only for a sensitivity analysis. Join it on (`well_id`, `year`, `month`) to the monthly table,
or rebuild any window feature from it with `gwstress.rainfall.rainfall_windows(obs, rain)` after
renaming `precip_3x3_mm` to `precip_mm`. See `reports/chirps_sampling_check.md`: overall agreement
is r = 0.999, and the 13 wells that differ a lot are all in steep terrain.
