-- ============================================================================
-- Stage 5: Expanding and Rolling-10 Baselines, Anomalies, and Completeness
-- Target: gw_stress.stage5_baselines
-- Minimum History Rule: K = 5
-- Standard Deviation Floor: 0.10 m
-- Cap: +/- 5.0
-- ============================================================================

CREATE OR REPLACE FUNCTION `river-data-511316-j4.gw_stress.calc_roll10_stats`(history ARRAY<FLOAT64>)
RETURNS STRUCT<n_roll10 INT64, mean_m FLOAT64, std_m FLOAT64>
LANGUAGE js AS R"""
  if (!history || history.length === 0) {
    return {n_roll10: 0, mean_m: null, std_m: null};
  }
  let non_nulls = [];
  for (let i = 0; i < history.length; i++) {
    if (history[i] !== null && !isNaN(history[i])) {
      non_nulls.push(history[i]);
    }
  }
  let prior = non_nulls.slice(-10);
  let n = prior.length;
  if (n < 5) {
    return {n_roll10: n, mean_m: null, std_m: null};
  }
  let sum = 0.0;
  for (let i = 0; i < n; i++) sum += prior[i];
  let mean = sum / n;
  let ssq = 0.0;
  for (let i = 0; i < n; i++) {
    let diff = prior[i] - mean;
    ssq += diff * diff;
  }
  let std = Math.sqrt(ssq / (n - 1));
  return {n_roll10: n, mean_m: mean, std_m: std};
""";

CREATE OR REPLACE TABLE `river-data-511316-j4.gw_stress.stage5_baselines` AS
WITH base_with_arrays AS (
  SELECT
    *,
    -- Expanding count of prior non-null readings in the same season
    COUNT(gwl_m_bgl) OVER (
      PARTITION BY well_id, season ORDER BY date
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS n_prior_same_season,
    -- Expanding mean
    AVG(gwl_m_bgl) OVER (
      PARTITION BY well_id, season ORDER BY date
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS raw_baseline_mean,
    -- Expanding std (sample std, ddof=1)
    STDDEV_SAMP(gwl_m_bgl) OVER (
      PARTITION BY well_id, season ORDER BY date
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS raw_baseline_std,
    -- Whole-record season presence
    LOGICAL_OR(gwl_m_bgl IS NOT NULL) OVER (
      PARTITION BY well_id, season
    ) AS has_season_obs,
    -- Completeness
    COUNT(gwl_m_bgl) OVER (
      PARTITION BY well_id ORDER BY date
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prior_obs_count,
    COUNT(1) OVER (
      PARTITION BY well_id ORDER BY date
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prior_round_count,
    COUNT(gwl_m_bgl) OVER (
      PARTITION BY well_id
    ) / 92.0 AS well_completeness_full_period,
    -- Array of all earlier readings for rolling-10 window
    ARRAY_AGG(gwl_m_bgl) OVER (
      PARTITION BY well_id, season ORDER BY date
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prior_readings_arr
  FROM `river-data-511316-j4.gw_stress.stage4_rain`
),
computed_baselines AS (
  SELECT
    *,
    `river-data-511316-j4.gw_stress.calc_roll10_stats`(prior_readings_arr) AS roll10
  FROM base_with_arrays
)
SELECT
  well_id,
  date,
  year,
  month,
  season,
  period_label,
  gwl_m_bgl,
  state,
  district,
  station_name,
  lat,
  lon,
  well_type,
  aquifer_type,
  well_depth_m,
  specific_yield,
  flag_zero,
  flag_extreme,
  flag_negative,
  flag_deeper_than_well,
  rain_1m_mm,
  rain_3m_mm,
  rain_6m_mm,
  n_prior_same_season,
  -- Expanding baseline (K=5 gate)
  CASE WHEN n_prior_same_season >= 5 THEN raw_baseline_mean ELSE NULL END AS baseline_mean_m,
  CASE WHEN n_prior_same_season >= 5 THEN raw_baseline_std ELSE NULL END AS baseline_std_m,
  CASE WHEN n_prior_same_season >= 5 AND raw_baseline_std < 0.10 THEN TRUE ELSE FALSE END AS flag_std_floored,
  -- Anomaly z
  CASE 
    WHEN n_prior_same_season >= 5 AND gwl_m_bgl IS NOT NULL THEN
      (gwl_m_bgl - raw_baseline_mean) / GREATEST(raw_baseline_std, 0.10)
    ELSE NULL
  END AS anomaly_z,
  CASE 
    WHEN n_prior_same_season >= 5 AND gwl_m_bgl IS NOT NULL THEN
      LEAST(GREATEST((gwl_m_bgl - raw_baseline_mean) / GREATEST(raw_baseline_std, 0.10), -5.0), 5.0)
    ELSE NULL
  END AS anomaly_z_capped,
  NOT has_season_obs AS well_has_no_series_for_season,
  -- Rolling-10 baseline
  roll10.n_roll10,
  roll10.mean_m AS baseline_roll10_mean_m,
  roll10.std_m AS baseline_roll10_std_m,
  CASE WHEN roll10.std_m IS NOT NULL AND roll10.std_m < 0.10 THEN TRUE ELSE FALSE END AS flag_roll10_std_floored,
  CASE 
    WHEN roll10.mean_m IS NOT NULL AND gwl_m_bgl IS NOT NULL THEN
      (gwl_m_bgl - roll10.mean_m) / GREATEST(roll10.std_m, 0.10)
    ELSE NULL
  END AS anomaly_roll10,
  CASE 
    WHEN roll10.mean_m IS NOT NULL AND gwl_m_bgl IS NOT NULL THEN
      LEAST(GREATEST((gwl_m_bgl - roll10.mean_m) / GREATEST(roll10.std_m, 0.10), -5.0), 5.0)
    ELSE NULL
  END AS anomaly_roll10_capped,
  -- Completeness
  CASE WHEN prior_round_count > 0 THEN prior_obs_count / prior_round_count ELSE NULL END AS completeness_to_date,
  well_completeness_full_period
FROM computed_baselines;

-- Assertions
ASSERT (SELECT COUNT(*) FROM `river-data-511316-j4.gw_stress.stage5_baselines`) = 253828
  AS 'stage5_baselines row count must match 253,828';
ASSERT (SELECT COUNTIF(anomaly_z IS NOT NULL) FROM `river-data-511316-j4.gw_stress.stage5_baselines`) = 166306
  AS 'anomaly_z non-null count must be exactly 166,306';
ASSERT (SELECT COUNTIF(well_has_no_series_for_season) FROM `river-data-511316-j4.gw_stress.stage5_baselines`) = 10212
  AS 'well_has_no_series_for_season count must match 444 wells x 23 years = 10,212';
