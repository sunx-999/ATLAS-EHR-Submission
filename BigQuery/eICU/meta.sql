CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.meta` AS

WITH diagnosis_dedup AS (
  SELECT DISTINCT
    patientunitstayid,
    timestamp,
    label,
    NULLIF(TRIM(CAST(icd9code AS STRING)), '') AS icd9code
  FROM `mimic-iv-484300.eICU_processed.diagnosis_labels`
  WHERE patientunitstayid IS NOT NULL
    AND label IS NOT NULL
    AND TRIM(CAST(label AS STRING)) != ''
),

diagnosis_agg AS (
  SELECT
    patientunitstayid,

    ARRAY_AGG(
      STRUCT(
        timestamp,
        label,
        icd9code
      )
      ORDER BY timestamp, label, icd9code
    ) AS diagnoses,

    ARRAY_AGG(DISTINCT label IGNORE NULLS ORDER BY label) AS diagnosis_labels,

    ARRAY_AGG(DISTINCT icd9code IGNORE NULLS ORDER BY icd9code) AS icd9codes,

    COUNT(*) AS n_diagnosis_rows,
    COUNT(DISTINCT label) AS n_diagnosis_labels,
    COUNT(DISTINCT icd9code) AS n_icd9codes

  FROM diagnosis_dedup
  GROUP BY patientunitstayid
)

SELECT
  p.uniquepid,
  p.patienthealthsystemstayid,
  p.patientunitstayid,
  p.gender,
  p.age,
  p.hospitaladmitoffset,
  p.hospitaldischargeoffset,
  p.unitdischargeoffset,
  p.unitstaytype,
  p.icu_mortality,
  p.hospital_mortality,

  a.diagnoses,
  a.diagnosis_labels,
  a.icd9codes,

  COALESCE(a.n_diagnosis_rows, 0) AS n_diagnosis_rows,
  COALESCE(a.n_diagnosis_labels, 0) AS n_diagnosis_labels,
  COALESCE(a.n_icd9codes, 0) AS n_icd9codes

FROM `mimic-iv-484300.eICU_processed.patient_base` p
LEFT JOIN diagnosis_agg a
  ON p.patientunitstayid = a.patientunitstayid;