CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.vitalaperiodic_state_core` AS

WITH raw AS (
  SELECT
    patientunitstayid,
    observationoffset,
    noninvasivesystolic,
    noninvasivediastolic,
    noninvasivemean,
    paop,
    cardiacoutput,
    cardiacinput,
    svr,
    svri,
    pvr,
    pvri
  FROM `physionet-data.eicu_crd.vitalaperiodic`
),

long_format AS (
  SELECT
    patientunitstayid,
    observationoffset AS timestamp,
    label,
    CAST(value AS STRING) AS value
  FROM raw
  UNPIVOT (
    value FOR label IN (
      noninvasivesystolic,
      noninvasivediastolic,
      noninvasivemean,
      paop,
      cardiacoutput,
      cardiacinput,
      svr,
      svri,
      pvr,
      pvri
    )
  )
)

SELECT
  patientunitstayid,
  timestamp,
  label,
  value
FROM long_format
WHERE patientunitstayid IS NOT NULL
  AND timestamp IS NOT NULL
  AND label IS NOT NULL
  AND value IS NOT NULL
  AND TRIM(value) != '';