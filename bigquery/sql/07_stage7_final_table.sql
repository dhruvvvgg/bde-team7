-- Stage 7: Final partitioned / clustered features table (84 columns)
-- Matching reference schema, column order, and types exactly.
-- Clustered by state and well_id for optimal cross-state validation queries.

CREATE OR REPLACE TABLE `river-data-511316-j4.gw_stress.final_features`
CLUSTER BY state, well_id AS
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
  baseline_mean_m,
  baseline_std_m,
  flag_std_floored,
  anomaly_z,
  anomaly_z_capped,
  well_has_no_series_for_season,
  n_roll10,
  baseline_roll10_mean_m,
  baseline_roll10_std_m,
  flag_roll10_std_floored,
  anomaly_roll10,
  anomaly_roll10_capped,
  anomaly_tadj,
  anomaly_tadj_capped,
  trend_5y_m_per_yr,
  rain_1m_normal_mm,
  rain_1m_dev_mm,
  rain_1m_dev_pct,
  rain_3m_normal_mm,
  rain_3m_dev_mm,
  rain_3m_dev_pct,
  rain_6m_normal_mm,
  rain_6m_dev_mm,
  rain_6m_dev_pct,
  completeness_to_date,
  well_completeness_full_period,
  gw_prev_round_m,
  gw_prev_change_m,
  gw_delta_from_prev_round_m,
  gw_same_season_lag1y_m,
  gw_same_season_lag2y_m,
  gw_max_to_date_m,
  gw_min_to_date_m,
  n_readings_to_date,
  gw_last_obs_m,
  rounds_since_last_obs,
  gw_fluct_prev_wy_m,
  rain_12m_mm,
  last_monsoon_year,
  rain_last_monsoon_mm,
  rain_last_monsoon_dev_mm,
  rain_last_monsoon_dev_pct,
  rain_3m_z,
  rain_3m_z_capped,
  rain_6m_z,
  rain_6m_z_capped,
  last_ne_year,
  rain_last_ne_mm,
  rain_last_ne_dev_mm,
  rain_last_ne_z_capped,
  rain_monsoon_ytd_mm,
  water_year,
  water_year_index,
  season_order,
  stress_category,
  stress_roll10,
  stress_tadj,
  suspect_reading,
  suspect_reason,
  state_slug
FROM `river-data-511316-j4.gw_stress.stage6_stress`;

-- Integrity assertions
ASSERT (SELECT COUNT(*) FROM `river-data-511316-j4.gw_stress.final_features`) = 253828
  AS "final_features must have exactly 253,828 rows";

ASSERT (SELECT COUNT(DISTINCT CONCAT(well_id, ':', period_label)) FROM `river-data-511316-j4.gw_stress.final_features`) = 253828
  AS "final_features must have unique (well_id, period_label)";

ASSERT (SELECT COUNT(DISTINCT state) FROM `river-data-511316-j4.gw_stress.final_features`) = 19
  AS "final_features must contain exactly 19 states";

ASSERT (SELECT COUNT(DISTINCT state_slug) FROM `river-data-511316-j4.gw_stress.final_features`) = 19
  AS "final_features must contain exactly 19 state slugs";
