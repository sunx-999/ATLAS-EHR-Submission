CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.vitalperiodic_state_core` AS

WITH raw AS (
  SELECT
    patientunitstayid,
    observationoffset,

    CAST(temperature AS FLOAT64) AS temperature,
    CAST(sao2 AS FLOAT64) AS sao2,
    CAST(heartrate AS FLOAT64) AS heartrate,
    CAST(respiration AS FLOAT64) AS respiration,
    CAST(cvp AS FLOAT64) AS cvp,
    CAST(etco2 AS FLOAT64) AS etco2,
    CAST(systemicsystolic AS FLOAT64) AS systemicsystolic,
    CAST(systemicdiastolic AS FLOAT64) AS systemicdiastolic,
    CAST(systemicmean AS FLOAT64) AS systemicmean,
    CAST(pasystolic AS FLOAT64) AS pasystolic,
    CAST(padiastolic AS FLOAT64) AS padiastolic,
    CAST(pamean AS FLOAT64) AS pamean,
    CAST(st1 AS FLOAT64) AS st1,
    CAST(st2 AS FLOAT64) AS st2,
    CAST(st3 AS FLOAT64) AS st3,
    CAST(icp AS FLOAT64) AS icp

  FROM `physionet-data.eicu_crd.vitalperiodic`
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
      temperature,
      sao2,
      heartrate,
      respiration,
      cvp,
      etco2,
      systemicsystolic,
      systemicdiastolic,
      systemicmean,
      pasystolic,
      padiastolic,
      pamean,
      st1,
      st2,
      st3,
      icp
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