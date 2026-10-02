# Handoff for modeling

This document describes the final table at `processed/gw_features_by_state/`.

**How to load it:**

```python
import pandas as pd

# whole table
df = pd.read_parquet("processed/gw_features_by_state")

# one state only
kl = pd.read_parquet("processed/gw_features_by_state",
                     filters=[("state_slug", "=", "kerala")])
```

**What one row is:** one well in one CGWB monitoring round. There are 2,759 wells and 92 rounds
(Jan, May, Aug and Nov from 2000 to 2022), giving 253,828 rows. Of these, 219,258 have a reading.

**Primary key:** (`well_id`, `date`).

## Column dictionary

Each column is assigned one of three classes:

- **A**: safe as a model input. It uses only information available at or before the round, and only
  rainfall from months before the reading.
- **B**: descriptive only. It uses future information, or it is an identifier, a QC artefact or a
  grouping helper. Do not feed it to a model.
- **C**: a candidate label or target.

| Column | Class | Description |
|---|:--:|---|
| `well_id` | B | Surrogate well key, a hash of state, district, station name and lat/lon. Use it for grouping and splits, never as a feature. `dataset/well_crosswalk.csv` maps it to the original codes. |
| `date` | B | First day of the monitoring month. CGWB does not publish the day of the reading. |
| `year` | A | Calendar year. Safe as a feature, but see the notes on splits below. |
| `month` | A | 1, 5, 8 or 11. |
| `season` | A | One of `post_monsoon_rabi` (Jan), `pre_monsoon` (May), `monsoon` (Aug) or `post_monsoon_kharif` (Nov). |
| `period_label` | B | The original column label, for example `May-07`. |
| `gwl_m_bgl` | C | Groundwater depth in metres below ground level. Larger means deeper, which means more stressed. It is the target for level forecasting. Using it as a feature for the same row leaks the target. |
| `state`, `district` | A/B | Standardised names. They are fine for grouping and splits. Using them as features undermines the cross-region test, so for held-out-state evaluation treat them as B. |
| `state_slug` | B | Partition key, a filesystem-safe version of `state`. |
| `station_name` | B | Identifier. |
| `lat`, `lon` | A | Well coordinates (static). |
| `well_type` | A | Dug, bore, tube or dug-cum-bore well (static). 94% are dug wells. |
| `aquifer_type` | A | Unconfined, semi-confined or confined (static). 127 wells are NA. |
| `well_depth_m` | A | Recorded well depth (static). |
| `specific_yield` | A | `Reference_Sy` from the authors: 5 values derived from aquifer classes on the hydrogeological map. Treat it as an ordinal or categorical aquifer-class covariate rather than a measured value. |
| `flag_zero` | B | The reading is exactly 0.0 m. There are 11 of these, and 3 look suspicious (see `reports/data_quality.md`). |
| `flag_extreme` | B | Modified z above 3.5 within the well-season, computed over the whole record, so it uses future information. QC only. |
| `flag_negative` | B | Negative depth. There are none. |
| `flag_deeper_than_well` | B | The reading is deeper than the recorded well depth (537 rows). QC only. |
| `rain_1m_mm`, `rain_3m_mm`, `rain_6m_mm` | A | CHIRPS rainfall totals over the 1, 3 or 6 complete months ending at M−1. Month M is never included. |
| `rain_{1,3,6}m_normal_mm` | A | That well's mean of the same window and season over **prior years only**. It needs at least 5 prior years. |
| `rain_{1,3,6}m_dev_mm` | A | Window total minus the normal. |
| `rain_{1,3,6}m_dev_pct` | A | Percentage deviation from the normal. It is NaN where the normal is below 5 mm. |
| `n_prior_same_season` | A | Number of earlier readings in the same well and season. |
| `baseline_mean_m`, `baseline_std_m` | A | Expanding mean and standard deviation of earlier same-season readings. NaN when there are fewer than 5 earlier readings (K = 5). |
| `flag_std_floored` | A | The baseline std was below 0.10 m and was floored to 0.10 m (251 rows). |
| `anomaly_z` | C | (reading − baseline mean) / max(std, 0.10). It uses the current reading. Raw values are heavy-tailed, ranging from −56 to +39. |
| `anomaly_z_capped` | C | `anomaly_z` clipped to ±5. **Use this one in models.** It is a target candidate, or an input only when lagged from an earlier round. |
| `well_has_no_series_for_season` | B | The well has no reading at all in this season over 2000–2022. It applies to May for 444 wells and uses the whole record. Use it only to group or exclude rows. |
| `trend_5y_m_per_yr` | A | Theil-Sen slope over the same season's readings from years Y−5 to Y−1. It needs at least 4 of those 5 readings and never includes the current reading. |
| `completeness_to_date` | A | Share of the well's earlier rounds that have a reading. |
| `well_completeness_full_period` | B | Share of all 92 rounds with a reading. It uses future information. |
| `stress_category` | C | One of `No reading`, `Insufficient history`, `Normal`, `Watch`, `Moderate Stress` or `High Stress` (default thresholds 1, 1.5 and 2 SD). Train and evaluate only on the last four. |

### Lagged targets as inputs

To use a C column as an input, lag it within the well. For example, take
`anomaly_z_capped` from the previous round, or from the same season last year, by grouping
on `well_id` and shifting after sorting by `date`. Its values at the current row describe the
current reading.

## Known data limitations

| Issue | Size | Consequence |
|---|---|---|
| May 2020 and May 2021 almost empty | 0.9% of wells measured in each year (COVID-19 period) | For these rounds there is nothing to train or test on. The May baseline does not update across them, and May 2022 trends are mostly NaN (2,014 of 2,038 readings). |
| Aug 2012 almost empty | 2.5% of wells measured | There is effectively no monsoon label for 2012. |
| Jan 2016 sparse | 19.1% of wells measured | This round has few labels, and they are not representative of all states. |
| Other rounds below 70% | Aug 2020 (58.5%), May 2016 (59.0%), Nov 2018 (66.7%) | Check how readings are spread across states before using these rounds as a test year. |
| No May series for 444 wells | All wells in Kerala (171) and West Bengal (68), 189 of 191 in Odisha, all 16 in Assam | These states have no pre-monsoon labels at all. All of their May rows are `No reading`. |
| First K readings of each well-season have no baseline | 52,952 readings (24.2%) at K = 5, mostly 2000–2004 | Labels effectively start around 2005. |
| `Station Code` corrupted upstream | All wells | Use `well_id`. Joining to other CGWB data needs name and lat/lon matching. |
| Monitoring date known only to the month | All rows | Rainfall windows stop at M−1. Do not add same-month rainfall. |

## How these should shape the splits

1. **Split by state, with whole states held out.** This matches the project question, and the data
   is already partitioned by `state_slug`. Never let one well appear in both train and test. If you
   split below the state level, use `GroupKFold` on `well_id`.
2. **Keep the four no-May states out of any May-specific evaluation.** For an all-season model,
   report per-season metrics. Pre-monsoon results for held-out Kerala, West Bengal, Odisha or Assam
   cannot be computed, and that should be stated rather than shown as zero.
3. **Temporal splits.** Train on earlier years and test on later ones, for example train up to 2017
   and test on 2018–2022. Do not choose 2020 or 2021 as the only test years for pre-monsoon, and do
   not use 2012 for monsoon. Exclude 2000–2004 from evaluation, because labels there are mostly
   `Insufficient history`.
4. **Exclude rows** whose `stress_category` is `No reading` or `Insufficient history` from training
   and evaluation of the classifier. Report how many were excluded.
5. **Class imbalance.** Under the default thresholds, classified rows break down as follows:

   | Category | Rows | Share of classified rows |
   |---|---:|---:|
   | Normal | 135,523 | 81.5% |
   | Watch | 12,661 | 7.6% |
   | Moderate Stress | 7,513 | 4.5% |
   | High Stress | 10,609 | 6.4% |

   Use stratified metrics such as macro-F1 and per-class recall, not plain accuracy.
6. **Rerun headline results under at least one other threshold set.** The threshold set changes
   8–23% of classified labels (see `reports/stage6_stress.md`).
7. **Never** use class B columns as inputs. Recompute rainfall normals or baselines inside a fold only
   if your split could change which data counts as "prior". The current columns are already strictly
   past-only per well, so they do not leak across a temporal split.
