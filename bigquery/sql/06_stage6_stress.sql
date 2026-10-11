-- Stage 6: Groundwater stress classification and data validity flags
-- Categories: No reading > Insufficient history > High Stress (>=2) > Moderate Stress (>=1.5) > Watch (>=1) > Normal
-- Suspect reasons: zero_suspicious, deeper_than_well, extreme_anomaly (|z| > 5)

CREATE OR REPLACE FUNCTION `river-data-511316-j4.gw_stress.classify_stress`(gwl FLOAT64, z FLOAT64)
RETURNS STRING AS (
  CASE
    WHEN gwl IS NULL THEN 'No reading'
    WHEN z IS NULL THEN 'Insufficient history'
    WHEN z >= 2.0 THEN 'High Stress'
    WHEN z >= 1.5 THEN 'Moderate Stress'
    WHEN z >= 1.0 THEN 'Watch'
    ELSE 'Normal'
  END
);

CREATE OR REPLACE TABLE `river-data-511316-j4.gw_stress.stage6_stress` AS
WITH base AS (
  SELECT
    *,
    -- Three suspicious zeros documented in config.SUSPICIOUS_ZEROS
    CONCAT(well_id, ':', period_label) IN ('Waf941bb407:Aug-19', 'W50b118a710:Nov-19', 'W9934cc4de3:Jan-18') AS r_zero,
    flag_deeper_than_well AS r_deep,
    COALESCE(ABS(anomaly_z) > 5.0, FALSE) AS r_ext
  FROM `river-data-511316-j4.gw_stress.stage5_all_features`
)
SELECT
  * EXCEPT(r_zero, r_deep, r_ext),
  `river-data-511316-j4.gw_stress.classify_stress`(gwl_m_bgl, anomaly_z) AS stress_category,
  `river-data-511316-j4.gw_stress.classify_stress`(gwl_m_bgl, anomaly_roll10) AS stress_roll10,
  `river-data-511316-j4.gw_stress.classify_stress`(gwl_m_bgl, anomaly_tadj) AS stress_tadj,
  (r_zero OR r_deep OR r_ext) AS suspect_reading,
  CASE
    WHEN r_zero AND r_deep THEN 'zero_suspicious;deeper_than_well'
    WHEN r_zero THEN 'zero_suspicious'
    WHEN r_deep AND r_ext THEN 'deeper_than_well;extreme_anomaly'
    WHEN r_deep THEN 'deeper_than_well'
    WHEN r_ext THEN 'extreme_anomaly'
    ELSE NULL
  END AS suspect_reason,
  LOWER(REGEXP_REPLACE(state, r'[^a-zA-Z0-9]+', '_')) AS state_slug
FROM base;

-- Assertions
ASSERT (SELECT COUNT(*) FROM `river-data-511316-j4.gw_stress.stage6_stress`) = 253828
  AS "stage6_stress must have exactly 253,828 rows";

-- Check that capped anomaly gives exact same categories
ASSERT (SELECT COUNTIF(stress_category != `river-data-511316-j4.gw_stress.classify_stress`(gwl_m_bgl, anomaly_z_capped)) FROM `river-data-511316-j4.gw_stress.stage6_stress`) = 0
  AS "stress_category must be identical under raw and capped anomaly_z";

ASSERT (SELECT COUNTIF(stress_roll10 != `river-data-511316-j4.gw_stress.classify_stress`(gwl_m_bgl, anomaly_roll10_capped)) FROM `river-data-511316-j4.gw_stress.stage6_stress`) = 0
  AS "stress_roll10 must be identical under raw and capped anomaly_roll10";

ASSERT (SELECT COUNTIF(stress_tadj != `river-data-511316-j4.gw_stress.classify_stress`(gwl_m_bgl, anomaly_tadj_capped)) FROM `river-data-511316-j4.gw_stress.stage6_stress`) = 0
  AS "stress_tadj must be identical under raw and capped anomaly_tadj";

-- Check suspect counts
ASSERT (SELECT COUNTIF(suspect_reason LIKE '%zero_suspicious%') FROM `river-data-511316-j4.gw_stress.stage6_stress`) = 3
  AS "exactly 3 suspicious zero rows must be flagged";
