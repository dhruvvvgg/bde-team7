-- ============================================================================
-- Stage 3: Cleaning, Standardization, and QC Flags
-- Target: gw_stress.stage3_clean
-- Rows expected: 253,828 (same as stage 2; no readings or rows removed)
-- ============================================================================

CREATE OR REPLACE FUNCTION `river-data-511316-j4.gw_stress.standardise_name`(name STRING)
RETURNS STRING
LANGUAGE js AS R"""
  if (!name) return null;
  let x = name.replace(/\s+/g, ' ').trim();
  x = x.replace(/\s*&\s*/g, ' and ');
  let words = x.split(' ');
  for (let i = 0; i < words.length; i++) {
    let w = words[i];
    if (w.toLowerCase() === 'and') {
      words[i] = 'and';
    } else if (w.length > 0) {
      words[i] = w.charAt(0).toUpperCase() + w.slice(1).toLowerCase();
    }
  }
  return words.join(' ');
""";

CREATE OR REPLACE TABLE `river-data-511316-j4.gw_stress.stage3_clean` AS
WITH cleaned_step1 AS (
  SELECT
    well_id,
    date,
    year,
    month,
    season,
    period_label,
    -- Placeholders (<= -900 or non-finite) to NULL
    CASE 
      WHEN gwl_m_bgl <= -900.0 OR IS_NAN(gwl_m_bgl) THEN NULL
      ELSE gwl_m_bgl
    END AS gwl_m_bgl,
    `river-data-511316-j4.gw_stress.standardise_name`(state) AS state,
    `river-data-511316-j4.gw_stress.standardise_name`(district) AS district,
    station_name,
    lat,
    lon,
    well_type,
    -- '-' in Aquifer Type represents unrecorded
    CASE WHEN aquifer_type = '-' THEN NULL ELSE aquifer_type END AS aquifer_type,
    well_depth_m,
    specific_yield
  FROM `river-data-511316-j4.gw_stress.stage2_long`
),
with_median AS (
  SELECT
    *,
    PERCENTILE_CONT(gwl_m_bgl, 0.5) OVER (PARTITION BY well_id, season) AS season_med
  FROM cleaned_step1
),
with_mad AS (
  SELECT
    *,
    PERCENTILE_CONT(ABS(gwl_m_bgl - season_med), 0.5) OVER (PARTITION BY well_id, season) AS season_mad
  FROM with_median
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
  -- Stage 3 QC Flags
  COALESCE(gwl_m_bgl = 0.0, FALSE) AS flag_zero,
  COALESCE(
    ABS(0.6745 * (gwl_m_bgl - season_med) / NULLIF(season_mad, 0.0)) > 3.5,
    FALSE
  ) AS flag_extreme,
  COALESCE(gwl_m_bgl < 0.0, FALSE) AS flag_negative,
  COALESCE(gwl_m_bgl > well_depth_m, FALSE) AS flag_deeper_than_well
FROM with_mad;

-- Asserts
ASSERT (SELECT COUNT(*) FROM `river-data-511316-j4.gw_stress.stage3_clean`) = 253828
  AS 'stage3_clean row count must match 253,828';
ASSERT (SELECT COUNTIF(flag_zero) FROM `river-data-511316-j4.gw_stress.stage3_clean`) = 11
  AS 'flag_zero count must be exactly 11';
ASSERT (SELECT COUNTIF(flag_negative) FROM `river-data-511316-j4.gw_stress.stage3_clean`) = 0
  AS 'flag_negative count must be exactly 0';
ASSERT (SELECT COUNTIF(flag_deeper_than_well) FROM `river-data-511316-j4.gw_stress.stage3_clean`) = 537
  AS 'flag_deeper_than_well count must be exactly 537';
