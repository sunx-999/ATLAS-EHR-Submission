CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.EHR_trimmed`
CLUSTER BY patientunitstayid, source, label AS

WITH per_icu AS (
  SELECT
    patientunitstayid,

    COUNT(*) AS n_events,
    COUNT(DISTINCT timestamp) AS n_unique_timestamps,

    MIN(timestamp) AS first_timestamp,
    MAX(timestamp) AS last_timestamp,
    MAX(timestamp) - MIN(timestamp) AS observed_span_min

  FROM `mimic-iv-484300.eICU_processed.EHR`
  GROUP BY patientunitstayid
),

keep_icu AS (
  SELECT
    patientunitstayid
  FROM per_icu
  WHERE n_unique_timestamps >= 5
    AND n_events >= 10
    AND n_unique_timestamps <= 4393
    AND observed_span_min <= 27535
)

SELECT
  e.*
FROM `mimic-iv-484300.eICU_processed.EHR` e
JOIN keep_icu k
  ON e.patientunitstayid = k.patientunitstayid;