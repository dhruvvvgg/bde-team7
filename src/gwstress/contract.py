"""Data contract for the final feature table (and the extension table).

This is the single source of truth for each column's class, dtype, allowed
range and meaning. tests/test_data_contract.py checks the data against it, and
scripts/11_handoff_dictionary.py writes the column dictionary in
docs/handoff_for_modeling.md from it, so the docs and the data cannot drift
apart.

Classes:
  A: safe model input (uses only earlier rounds, rainfall up to M-1, or static
     attributes).
  B: descriptive only (identifier, QC flag, whole-record statistic, or uses
     future information). Never a model input.
  C: label or target candidate (uses the current reading). Usable as an input
     only when lagged from an earlier round.

Dtype kinds: f float, i integer, b bool, s string, c category, M datetime.
`nan` is the policy for missing values: "none" (no NaN allowed) or a short
reason why NaN can occur.
Ranges are physical or definitional bounds, not the observed min/max, so
they stay valid for the extension and future reruns.
"""
from __future__ import annotations

INF = float("inf")

# (name, class, kind, nan_policy, lo, hi, description)
COLUMNS: list[tuple] = [
    # --- keys / identifiers ---
    ("well_id", "B", "s", "none", None, None, "Surrogate well key (hash of state, district, station name, lat, lon). Key, never a feature; `dataset/well_crosswalk.csv` maps it to the original codes."),
    ("date", "B", "M", "none", None, None, "First day of the monitoring month (the day is not published). Part of the key."),
    ("period_label", "B", "s", "none", None, None, "Original round label, e.g. `May-07`. Part of the key in `processed/splits.parquet`."),
    ("year", "A", "i", "none", 2000, 2024, "Calendar year."),
    ("month", "A", "i", "none", 1, 11, "Round month: 1, 5, 8 or 11."),
    ("season", "A", "c", "none", None, None, "`post_monsoon_rabi` (Jan), `pre_monsoon` (May), `monsoon` (Aug), `post_monsoon_kharif` (Nov)."),
    ("water_year", "A", "i", "none", 1999, 2024, "Start year of the Jun-May hydrological year that contains the round."),
    ("water_year_index", "A", "i", "none", 0, 25, "`water_year - 1999`."),
    ("season_order", "A", "i", "none", 0, 3, "Position within the water year: Aug=0, Nov=1, Jan=2, May=3."),
    # --- target / current reading ---
    ("gwl_m_bgl", "C", "f", "no reading in this round", 0.0, 200.0, "Depth to groundwater, m below ground level (larger = deeper = more stressed). Target for level forecasting."),
    # --- static covariates ---
    ("state", "A", "s", "none", None, None, "Standardised state name. Use for splits; as a feature it weakens the cross-region test (treat as B for held-out-state evaluation)."),
    ("state_slug", "B", "c", "none", None, None, "Partition key (file-system-safe state)."),
    ("district", "A", "s", "none", None, None, "Standardised district name (same caveat as `state`)."),
    ("station_name", "B", "s", "none", None, None, "Station name (identifier)."),
    ("lat", "A", "f", "none", 6.0, 37.5, "Latitude (WGS84)."),
    ("lon", "A", "f", "none", 68.0, 97.5, "Longitude (WGS84)."),
    ("well_type", "A", "s", "none", None, None, "Dug / bore / tube / dug-cum-bore well."),
    ("aquifer_type", "A", "s", "not recorded by CGWB (127 wells)", None, None, "Unconfined / semi-confined / confined."),
    ("well_depth_m", "A", "f", "none", 0.0, 400.0, "Recorded well depth (m). May be outdated (see `suspect_reading`)."),
    ("specific_yield", "A", "f", "none", 0.0, 0.2, "Authors' `Reference_Sy`: 5 aquifer-class values from the hydrogeological map. Treat as ordinal or categorical."),
    # --- stage-3 QC flags ---
    ("flag_zero", "B", "b", "none", None, None, "Reading is exactly 0.0 m."),
    ("flag_extreme", "B", "b", "none", None, None, "Modified z > 3.5 within the well-season over the WHOLE record (uses future data). QC only."),
    ("flag_negative", "B", "b", "none", None, None, "Negative depth (none occur)."),
    ("flag_deeper_than_well", "B", "b", "none", None, None, "Reading deeper than the recorded well depth (usually an outdated depth record)."),
    ("suspect_reading", "B", "b", "none", None, None, "Composite QC flag (Part D). Do not drop by default; use for an exclusion sensitivity run."),
    ("suspect_reason", "B", "s", "not suspect", None, None, "`zero_suspicious`, `deeper_than_well`, `extreme_anomaly`, `zero_extension` (extension only), joined with `;`."),
    # --- rainfall windows (stage 4) ---
    ("rain_1m_mm", "A", "f", "none", 0.0, 6000.0, "CHIRPS rainfall in month M-1 (nearest 0.05 deg cell)."),
    ("rain_3m_mm", "A", "f", "none", 0.0, 12000.0, "Rainfall over months M-3..M-1."),
    ("rain_6m_mm", "A", "f", "none", 0.0, 20000.0, "Rainfall over months M-6..M-1."),
    ("rain_1m_normal_mm", "A", "f", "< 5 earlier years (2000-2004)", 0.0, 6000.0, "Mean of `rain_1m_mm` for the same well and season over earlier years only."),
    ("rain_3m_normal_mm", "A", "f", "< 5 earlier years (2000-2004)", 0.0, 12000.0, "Same, 3-month window."),
    ("rain_6m_normal_mm", "A", "f", "< 5 earlier years (2000-2004)", 0.0, 20000.0, "Same, 6-month window."),
    ("rain_1m_dev_mm", "A", "f", "no normal", -INF, INF, "`rain_1m_mm` minus its earlier-years normal."),
    ("rain_3m_dev_mm", "A", "f", "no normal", -INF, INF, "Same, 3-month window."),
    ("rain_6m_dev_mm", "A", "f", "no normal", -INF, INF, "Same, 6-month window."),
    ("rain_1m_dev_pct", "A", "f", "no normal, or normal < 5 mm", -100.0, INF, "Percentage deviation from the normal."),
    ("rain_3m_dev_pct", "A", "f", "no normal, or normal < 5 mm", -100.0, INF, "Same, 3-month window."),
    ("rain_6m_dev_pct", "A", "f", "no normal, or normal < 5 mm", -100.0, INF, "Same, 6-month window."),
    ("rain_3m_z", "A", "f", "< 5 earlier years, or earlier std < 1 mm", -INF, INF, "z of `rain_3m_mm` against the same well and season in earlier years. Heavy-tailed: prefer the capped version."),
    ("rain_3m_z_capped", "A", "f", "as rain_3m_z", -5.0, 5.0, "`rain_3m_z` clipped to +/-5 (use in models)."),
    ("rain_6m_z", "A", "f", "< 5 earlier years, or earlier std < 1 mm", -INF, INF, "Same, 6-month window."),
    ("rain_6m_z_capped", "A", "f", "as rain_6m_z", -5.0, 5.0, "`rain_6m_z` clipped to +/-5 (use in models)."),
    ("rain_12m_mm", "A", "f", "Jan/May 2000 (CHIRPS starts 1999-07)", 0.0, 30000.0, "Rainfall over months M-12..M-1."),
    ("rain_monsoon_ytd_mm", "A", "f", "Jan/May 2000 (June 1999 missing)", 0.0, 30000.0, "Rainfall from 1 June of the current monsoon year up to M-1 (Jan: Jun-Dec; May: Jun-Apr; Aug: Jun-Jul; Nov: Jun-Oct)."),
    ("last_monsoon_year", "A", "i", "none", 1999, 2024, "Year of the last completed Jun-Sep monsoon (Y for Nov rounds, Y-1 otherwise)."),
    ("rain_last_monsoon_mm", "A", "f", "Jan/May/Aug 2000 (June 1999 missing)", 0.0, 20000.0, "Jun-Sep total of `last_monsoon_year`."),
    ("rain_last_monsoon_dev_mm", "A", "f", "< 5 earlier monsoons", -INF, INF, "That total minus the well's mean over earlier monsoons."),
    ("rain_last_monsoon_dev_pct", "A", "f", "< 5 earlier monsoons", -100.0, INF, "Same, in percent."),
    ("last_ne_year", "A", "i", "none", 1999, 2023, "Year of the last completed Oct-Dec (north-east) monsoon: always Y-1."),
    ("rain_last_ne_mm", "A", "f", "none", 0.0, 10000.0, "Oct+Nov+Dec rainfall of `last_ne_year`."),
    ("rain_last_ne_dev_mm", "A", "f", "< 5 earlier Oct-Dec seasons", -INF, INF, "That total minus the well's mean over earlier Oct-Dec seasons."),
    ("rain_last_ne_z_capped", "A", "f", "< 5 earlier seasons, or std < 1 mm", -5.0, 5.0, "z of the Oct-Dec total against earlier seasons, clipped to +/-5."),
    # --- expanding seasonal baseline (stage 5) ---
    ("n_prior_same_season", "A", "i", "none", 0, 30, "Number of earlier readings of the same well and season."),
    ("baseline_mean_m", "A", "f", "< K=5 earlier same-season readings", 0.0, 200.0, "Mean of all earlier same-season readings (expanding)."),
    ("baseline_std_m", "A", "f", "< K=5 earlier same-season readings", 0.0, 200.0, "Std (ddof=1) of the same. May be ~0; the anomaly floors it at 0.10 m."),
    ("flag_std_floored", "A", "b", "none", None, None, "`baseline_std_m` < 0.10 m and floored."),
    ("anomaly_z", "C", "f", "no reading, or no baseline", -INF, INF, "(reading - baseline_mean) / max(std, 0.10). Raw, heavy-tailed."),
    ("anomaly_z_capped", "C", "f", "as anomaly_z", -5.0, 5.0, "`anomaly_z` clipped to +/-5 (use in models)."),
    ("anomaly_tadj", "C", "f", "as anomaly_z", -40.0, 40.0, "Small-sample-adjusted normal-equivalent score: Phi^-1(F_t(t; n-1)), with t = (x - mean) / (s * sqrt(1 + 1/n))."),
    ("anomaly_tadj_capped", "C", "f", "as anomaly_z", -5.0, 5.0, "`anomaly_tadj` clipped to +/-5."),
    ("stress_category", "C", "c", "none", None, None, "PRIMARY label: No reading / Insufficient history / Normal / Watch / Moderate Stress / High Stress (thresholds 1 / 1.5 / 2 on `anomaly_z`)."),
    ("stress_tadj", "C", "c", "none", None, None, "Same labels on `anomaly_tadj` (sensitivity label)."),
    # --- rolling baseline (Part B) ---
    ("n_roll10", "A", "i", "none", 0, 10, "Number of earlier same-season readings in the rolling window (max 10)."),
    ("baseline_roll10_mean_m", "A", "f", "< 5 earlier same-season readings", 0.0, 200.0, "Mean of the last 10 earlier same-season readings."),
    ("baseline_roll10_std_m", "A", "f", "< 5 earlier same-season readings", 0.0, 200.0, "Std (ddof=1) of the same."),
    ("flag_roll10_std_floored", "A", "b", "none", None, None, "Rolling std < 0.10 m and floored."),
    ("anomaly_roll10", "C", "f", "no reading, or no rolling baseline", -INF, INF, "Anomaly against the rolling baseline (raw)."),
    ("anomaly_roll10_capped", "C", "f", "as anomaly_roll10", -5.0, 5.0, "Clipped to +/-5."),
    ("stress_roll10", "C", "c", "none", None, None, "Same labels on `anomaly_roll10` (sensitivity label)."),
    # --- trend, completeness ---
    ("trend_5y_m_per_yr", "A", "f", "< 4 of the 5 earlier same-season readings", -INF, INF, "Theil-Sen slope over the same season, years Y-5..Y-1 (m/yr)."),
    ("completeness_to_date", "A", "f", "first round of each well", 0.0, 1.0, "Share of the well's earlier rounds that have a reading."),
    ("well_completeness_full_period", "B", "f", "none", 0.0, 1.0, "Share of ALL rounds with a reading (uses future data)."),
    ("well_has_no_series_for_season", "B", "b", "none", None, None, "The well has no reading at all in this season over the whole record (uses future data; for grouping)."),
    # --- groundwater history (Part A) ---
    ("gw_prev_round_m", "A", "f", "previous round missing", 0.0, 200.0, "Reading of the immediately previous round."),
    ("gw_last_obs_m", "A", "f", "no earlier reading", 0.0, 200.0, "Most recent earlier non-missing reading."),
    ("rounds_since_last_obs", "A", "f", "no earlier reading", 1.0, 100.0, "How many rounds back `gw_last_obs_m` is."),
    ("gw_same_season_lag1y_m", "A", "f", "that round missing", 0.0, 200.0, "Same season, previous year."),
    ("gw_same_season_lag2y_m", "A", "f", "that round missing", 0.0, 200.0, "Same season, two years earlier."),
    ("gw_prev_change_m", "A", "f", "t-1 or t-2 missing", -INF, INF, "reading(t-1) - reading(t-2)."),
    ("gw_delta_from_prev_round_m", "C", "f", "t or t-1 missing", -INF, INF, "reading(t) - reading(t-1). Uses the current reading: target candidate, NOT an input."),
    ("gw_max_to_date_m", "A", "f", "no earlier reading", 0.0, 200.0, "Deepest earlier reading."),
    ("gw_min_to_date_m", "A", "f", "no earlier reading", 0.0, 200.0, "Shallowest earlier reading."),
    ("n_readings_to_date", "A", "i", "none", 0, 120, "Number of earlier non-missing readings."),
    ("gw_fluct_prev_wy_m", "A", "f", "May or Nov of Y-1 missing", -INF, INF, "gw(Nov, Y-1) - gw(May, Y-1): last completed monsoon's depth change (negative = recharge)."),
]

EXTENSION_ONLY = [("is_extension", "B", "b", "none", None, None, "True for every row of the 2023-2024 extension table.")]

KEY = ["well_id", "date"]
NO_NAN_RAINFALL = ["rain_1m_mm", "rain_3m_mm", "rain_6m_mm", "rain_last_ne_mm"]


def by_name() -> dict:
    return {c[0]: c for c in COLUMNS + EXTENSION_ONLY}
