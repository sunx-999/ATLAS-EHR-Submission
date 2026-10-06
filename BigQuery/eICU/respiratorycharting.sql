CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.respiratorycharting_state_core` AS

WITH raw AS (
  SELECT
    patientunitstayid,
    respchartoffset,
    respchartentryoffset,
    respcharttypecat,
    respchartvaluelabel,
    respchartvalue
  FROM `physionet-data.eicu_crd.respiratorycharting`
),

cleaned AS (
  SELECT *
  FROM raw
  WHERE patientunitstayid IS NOT NULL
    AND COALESCE(respchartoffset, respchartentryoffset) IS NOT NULL
    AND respchartvalue IS NOT NULL
    AND TRIM(respchartvalue) != ''
    AND NOT REGEXP_CONTAINS(
      LOWER(TRIM(respchartvalue)),
      r'^(not assessed|unable to assess|unable|not performed|refused|deferred|n/a|na|unknown|comment|see note|null|missing|documentation undone)$'
    )
),

state_core AS (
  SELECT *
  FROM cleaned
  WHERE
    (
      respcharttypecat = 'respFlowPtVentData'
      AND respchartvaluelabel IN (
        'RR (patient)',
        'Total RR',
        'SaO2',
        'ETCO2',
        'EtCO2',
        'Exhaled MV',
        'Exhaled TV (machine)',
        'Exhaled TV (patient)',
        'Mean Airway Pressure',
        'Peak Insp. Pressure',
        'Plateau Pressure',
        'Compliance',
        'HR'
      )
    )

    OR (
      respcharttypecat = 'respFlowCareData'
      AND respchartvaluelabel IN (
        'f Total',
        'Resp Rate Total',
        'RR Spont',
        'Spontaneous Respiratory Rate',
        'VS RESP RATE',

        'Adult Con Pt/Vent Spont Rate',
        'NIV Pt/Vent Spont_Rate',

        'Adult Con Pt/Vent SpO2',
        'NIV Pt/Vent SpO2_5',
        'PULSE OX RESULTS VT',

        'Adult Con Pt/Vent ETCO2',
        'Adult Con Pt/Vent ETCO2_',
        'ETCO2',
        'EtCO2',

        'Adult Con Pt/Vent MinuteVentil',
        'Measured Ve',
        'MVe',
        'Minute Volume, Spontaneous',

        'Adult Con Pt/Vent InspiratorTV',
        'Vti',
        'Tidal Volume Observed (VT)',
        'Tidal Volume, Delivered',
        'Exhaled Vt',
        'Spont TV',
        'Vt Spontaneous (mL)',
        'NIV Pt/Vent Spont_TidalV',

        'Mean Airway Pressure',
        'Peak Pressure',
        'Plateau Pressure',
        'Mechanical Ventilator Compliance',
        'Static Compliance',
        'Mechanical Ventilator Resistance',

        'Breath Sounds',
        'Cough Description',
        'Sputum Amount',
        'Sputum Color',
        'Sputum Consistency'
      )
    )
),

value_filtered AS (
  SELECT *
  FROM state_core
  WHERE NOT (
    respchartvaluelabel = 'Cough Description'
    AND LOWER(TRIM(respchartvalue)) IN (
      'refused',
      'refused to cough',
      'reported,not observed',
      'suctioned'
    )
  )
),

timestamp_resolved AS (
  SELECT
    patientunitstayid,

    CASE
      WHEN respchartoffset IS NOT NULL THEN respchartoffset
      WHEN respchartoffset IS NULL AND respchartentryoffset IS NOT NULL THEN respchartentryoffset
      ELSE NULL
    END AS timestamp,

    CONCAT(
      TRIM(respcharttypecat),
      ' | ',
      TRIM(respchartvaluelabel)
    ) AS label,

    TRIM(respchartvalue) AS value

  FROM value_filtered
),

final_filtered AS (
  SELECT *
  FROM timestamp_resolved
  WHERE patientunitstayid IS NOT NULL
    AND timestamp IS NOT NULL
    AND label IS NOT NULL
    AND TRIM(label) != ''
    AND value IS NOT NULL
    AND TRIM(value) != ''
    AND NOT REGEXP_CONTAINS(
      LOWER(TRIM(value)),
      r'^(n/a|na|null|none missing|missing|unknown|comment|see note|documentation undone)$'
    )
)

SELECT
  patientunitstayid,
  timestamp,
  label,
  value
FROM final_filtered;