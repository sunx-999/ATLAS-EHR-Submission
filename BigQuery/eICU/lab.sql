CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.lab_state_core` AS

WITH prepared AS (
  SELECT
    patientunitstayid,

    COALESCE(labresultoffset, labresultrevisedoffset) AS timestamp,

    TRIM(CAST(labname AS STRING)) AS labname,

    LOWER(TRIM(CAST(labname AS STRING))) AS labname_lower,

    COALESCE(
      NULLIF(TRIM(CAST(labmeasurenamesystem AS STRING)), ''),
      NULLIF(TRIM(CAST(labmeasurenameinterface AS STRING)), '')
    ) AS resolved_unit,

    COALESCE(
      NULLIF(TRIM(CAST(labresulttext AS STRING)), ''),
      NULLIF(TRIM(CAST(labresult AS STRING)), '')
    ) AS value

  FROM `physionet-data.eicu_crd.lab`
),

filtered AS (
  SELECT
    patientunitstayid,
    timestamp,

    CASE
      WHEN labname_lower = 'ph'
        THEN 'pH'
      WHEN labname_lower = 'anion gap'
        THEN 'anion gap | mmol/L'
      WHEN labname_lower = 'urinary specific gravity'
        THEN 'urinary specific gravity'
      WHEN resolved_unit IS NOT NULL
        THEN CONCAT(labname, ' | ', resolved_unit)
      ELSE labname
    END AS label,

    value

  FROM prepared
  WHERE patientunitstayid IS NOT NULL
    AND timestamp IS NOT NULL
    AND labname IS NOT NULL
    AND labname != ''
    AND value IS NOT NULL
    AND TRIM(value) != ''

    AND NOT REGEXP_CONTAINS(
      LOWER(TRIM(value)),
      r'^(n/a|na|null|none missing|missing|unknown|comment|see note)$'
    )

    AND (
      resolved_unit IS NOT NULL
      OR labname_lower IN (
        'ph',
        'anion gap',
        'urinary specific gravity'
      )
    )

    AND labname_lower NOT IN (
      'device',
      'mode',
      'site',
      'vent other',
      'tv',
      'vent rate',
      'peep',
      'fio2',
      'lpm o2'
    )
)

SELECT
  patientunitstayid,
  timestamp,
  label,
  value
FROM filtered;