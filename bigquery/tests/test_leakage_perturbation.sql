-- Leakage perturbation test: "Perturb the future"
-- Replaces all readings and rainfall values after cutoff year 2015 with extreme values.
-- Recomputes all features and asserts that no row before the cutoff year changed.

CREATE TEMP TABLE test_perturbed_gwl AS
SELECT
  well_id,
  period_label,
  date,
  year,
  month,
  season,
  state,
  CASE WHEN year >= 2015 THEN 99.99 ELSE gwl_m_bgl END AS gwl_m_bgl
FROM `river-data-511316-j4.gw_stress.stage2_long`
WHERE well_id = 'Wc0eb3debd6';

CREATE TEMP TABLE test_perturbed_rain AS
SELECT
  well_id,
  year,
  month,
  CASE WHEN year >= 2015 THEN 9999.0 ELSE precip_mm END AS precip_mm
FROM `river-data-511316-j4.gw_stress.chirps_monthly`
WHERE well_id = 'Wc0eb3debd6';

-- Compute features on perturbed data
CREATE TEMP TABLE test_perturbed_features AS
WITH
gw_raw_pos AS (
  SELECT
    well_id,
    period_label,
    date,
    gwl_m_bgl,
    ROW_NUMBER() OVER (PARTITION BY well_id ORDER BY date) - 1 AS pos
  FROM test_perturbed_gwl
),
gw_history AS (
  SELECT
    well_id,
    period_label,
    date,
    gwl_m_bgl,
    LAG(gwl_m_bgl, 1) OVER (PARTITION BY well_id ORDER BY date) AS gw_prev_round_m,
    LAG(gwl_m_bgl, 1) OVER (PARTITION BY well_id ORDER BY date) - LAG(gwl_m_bgl, 2) OVER (PARTITION BY well_id ORDER BY date) AS gw_prev_change_m,
    gwl_m_bgl - LAG(gwl_m_bgl, 1) OVER (PARTITION BY well_id ORDER BY date) AS gw_delta_from_prev_round_m,
    LAG(gwl_m_bgl, 4) OVER (PARTITION BY well_id ORDER BY date) AS gw_same_season_lag1y_m,
    LAG(gwl_m_bgl, 8) OVER (PARTITION BY well_id ORDER BY date) AS gw_same_season_lag2y_m,
    MAX(gwl_m_bgl) OVER (PARTITION BY well_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS gw_max_to_date_m,
    MIN(gwl_m_bgl) OVER (PARTITION BY well_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS gw_min_to_date_m,
    COUNT(gwl_m_bgl) OVER (PARTITION BY well_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_readings_to_date,
    LAST_VALUE(gwl_m_bgl IGNORE NULLS) OVER (PARTITION BY well_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS gw_last_obs_m,
    pos - LAST_VALUE(CASE WHEN gwl_m_bgl IS NOT NULL THEN pos END IGNORE NULLS)
      OVER (PARTITION BY well_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS rounds_since_last_obs
  FROM gw_raw_pos
),
gw_baselines AS (
  SELECT
    well_id,
    period_label,
    year,
    month,
    season,
    date,
    gwl_m_bgl,
    COUNT(gwl_m_bgl) OVER w AS n_prior_same_season,
    IF(COUNT(gwl_m_bgl) OVER w >= 5, AVG(gwl_m_bgl) OVER w, NULL) AS baseline_mean_m,
    IF(COUNT(gwl_m_bgl) OVER w >= 5, STDDEV_SAMP(gwl_m_bgl) OVER w, NULL) AS baseline_std_m
  FROM test_perturbed_gwl
  WINDOW w AS (
    PARTITION BY well_id, season
    ORDER BY year
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
  )
)
SELECT
  b.well_id,
  b.period_label,
  b.year,
  b.date,
  b.n_prior_same_season,
  b.baseline_mean_m,
  b.baseline_std_m,
  CASE WHEN b.n_prior_same_season >= 5 AND b.gwl_m_bgl IS NOT NULL THEN
    (b.gwl_m_bgl - b.baseline_mean_m) / GREATEST(b.baseline_std_m, 0.10)
  ELSE NULL END AS anomaly_z,
  h.gw_prev_round_m,
  h.gw_last_obs_m,
  h.rounds_since_last_obs,
  h.gw_max_to_date_m
FROM gw_baselines b
JOIN gw_history h ON b.well_id = h.well_id AND b.period_label = h.period_label;

-- Compare with final_features before cutoff year 2015
ASSERT (
  SELECT COUNT(*)
  FROM test_perturbed_features p
  JOIN `river-data-511316-j4.gw_stress.final_features` orig
    ON p.well_id = orig.well_id AND p.period_label = orig.period_label
  WHERE p.year < 2015
    AND (
      ABS(COALESCE(p.anomaly_z, 0) - COALESCE(orig.anomaly_z, 0)) > 1e-9 OR
      ABS(COALESCE(p.baseline_mean_m, 0) - COALESCE(orig.baseline_mean_m, 0)) > 1e-9 OR
      ABS(COALESCE(p.baseline_std_m, 0) - COALESCE(orig.baseline_std_m, 0)) > 1e-9 OR
      COALESCE(p.n_prior_same_season, 0) != COALESCE(orig.n_prior_same_season, 0) OR
      ABS(COALESCE(p.gw_prev_round_m, 0) - COALESCE(orig.gw_prev_round_m, 0)) > 1e-9 OR
      ABS(COALESCE(p.gw_last_obs_m, 0) - COALESCE(orig.gw_last_obs_m, 0)) > 1e-9 OR
      COALESCE(p.rounds_since_last_obs, 0) != COALESCE(orig.rounds_since_last_obs, 0) OR
      ABS(COALESCE(p.gw_max_to_date_m, 0) - COALESCE(orig.gw_max_to_date_m, 0)) > 1e-9
    )
) = 0
AS "Leakage test failed: Features before 2015 changed when future was perturbed!";
