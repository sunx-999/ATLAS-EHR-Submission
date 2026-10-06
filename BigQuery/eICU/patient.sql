CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.patient_base` AS

WITH base AS (
  SELECT
    uniquepid,
    patienthealthsystemstayid,
    patientunitstayid,

    NULLIF(TRIM(CAST(gender AS STRING)), '') AS gender_raw,
    NULLIF(TRIM(CAST(age AS STRING)), '') AS age_raw,

    hospitaladmitoffset,
    hospitaldischargeoffset,
    unitdischargeoffset,

    unitstaytype,

    unitdischargestatus,
    hospitaldischargestatus

  FROM `physionet-data.eicu_crd.patient`
  WHERE patientunitstayid IS NOT NULL
),

cleaned AS (
  SELECT
    uniquepid,
    patienthealthsystemstayid,
    patientunitstayid,

    CASE
      WHEN LOWER(gender_raw) = 'male' THEN 'Male'
      WHEN LOWER(gender_raw) = 'female' THEN 'Female'
      ELSE NULL
    END AS gender,

    CASE
      WHEN age_raw = '> 89' THEN 90
      WHEN SAFE_CAST(age_raw AS INT64) IS NOT NULL THEN SAFE_CAST(age_raw AS INT64)
      ELSE NULL
    END AS age,

    hospitaladmitoffset,
    hospitaldischargeoffset,
    unitdischargeoffset,

    unitstaytype,

    CASE
      WHEN LOWER(TRIM(CAST(unitdischargestatus AS STRING))) = 'expired' THEN 1
      WHEN unitdischargestatus IS NULL OR TRIM(CAST(unitdischargestatus AS STRING)) = '' THEN NULL
      ELSE 0
    END AS icu_mortality,

    CASE
      WHEN LOWER(TRIM(CAST(hospitaldischargestatus AS STRING))) = 'expired' THEN 1
      WHEN hospitaldischargestatus IS NULL OR TRIM(CAST(hospitaldischargestatus AS STRING)) = '' THEN NULL
      ELSE 0
    END AS hospital_mortality

  FROM base
)

SELECT
  uniquepid,
  patienthealthsystemstayid,
  patientunitstayid,
  gender,
  age,
  hospitaladmitoffset,
  hospitaldischargeoffset,
  unitdischargeoffset,
  unitstaytype,
  icu_mortality,
  hospital_mortality
FROM cleaned
WHERE age >= 18
  AND gender IN ('Male', 'Female');