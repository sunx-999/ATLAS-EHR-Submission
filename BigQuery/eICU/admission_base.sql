CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.admission_base` AS

WITH unit_base AS (
  SELECT
    uniquepid,
    patienthealthsystemstayid,
    patientunitstayid,

    gender,
    age,
    icu_mortality,
    hospital_mortality,

    SAFE_CAST(hospitaladmitoffset AS INT64) AS hospitaladmitoffset,
    SAFE_CAST(hospitaldischargeoffset AS INT64) AS hospitaldischargeoffset,
    SAFE_CAST(unitdischargeoffset AS INT64) AS unitdischargeoffset,

    COALESCE(n_diagnosis_rows, 0) AS n_diagnosis_rows,
    COALESCE(n_diagnosis_labels, 0) AS n_diagnosis_labels,
    COALESCE(n_icd9codes, 0) AS n_icd9codes

  FROM `mimic-iv-484300.eICU_processed.meta`
  WHERE patienthealthsystemstayid IS NOT NULL
    AND patientunitstayid IS NOT NULL
),

unit_derived AS (
  SELECT
    *,

    -hospitaladmitoffset AS icu_admit_min_from_hospital_admit,

    unitdischargeoffset - hospitaladmitoffset AS icu_discharge_min_from_hospital_admit,

    hospitaldischargeoffset - hospitaladmitoffset AS hospital_discharge_min_from_hospital_admit,

    SAFE_DIVIDE(unitdischargeoffset, 1440.0) AS icu_los_days,

    SAFE_DIVIDE(hospitaldischargeoffset - hospitaladmitoffset, 1440.0) AS hospital_los_days

  FROM unit_base
),

ordered_units AS (
  SELECT
    *,

    ROW_NUMBER() OVER (
      PARTITION BY patienthealthsystemstayid
      ORDER BY
        icu_admit_min_from_hospital_admit,
        icu_discharge_min_from_hospital_admit,
        patientunitstayid
    ) AS icu_order_within_admission,

    LAG(icu_discharge_min_from_hospital_admit) OVER (
      PARTITION BY patienthealthsystemstayid
      ORDER BY
        icu_admit_min_from_hospital_admit,
        icu_discharge_min_from_hospital_admit,
        patientunitstayid
    ) AS prev_icu_discharge_min_from_hospital_admit

  FROM unit_derived
),

unit_flags AS (
  SELECT
    *,

    hospitaladmitoffset IS NULL AS flag_missing_hospitaladmitoffset,
    hospitaldischargeoffset IS NULL AS flag_missing_hospitaldischargeoffset,
    unitdischargeoffset IS NULL AS flag_missing_unitdischargeoffset,

    hospitaladmitoffset > 0 AS flag_icu_admit_before_hospital_admit,
    unitdischargeoffset < 0 AS flag_icu_discharge_before_icu_admit,
    hospitaldischargeoffset <= hospitaladmitoffset AS flag_nonpositive_hospital_los,
    unitdischargeoffset > hospitaldischargeoffset AS flag_icu_discharge_after_hospital_discharge,

    prev_icu_discharge_min_from_hospital_admit
      > icu_admit_min_from_hospital_admit AS flag_icu_overlap_previous

  FROM ordered_units
),

admission_agg AS (
  SELECT
    patienthealthsystemstayid AS admission_id,
    ANY_VALUE(uniquepid) AS uniquepid,

    ANY_VALUE(gender) AS gender,
    ANY_VALUE(age) AS age,

    COUNT(*) AS n_icu_stays,
    COUNT(DISTINCT patientunitstayid) AS n_distinct_icu_stays,

    ARRAY_AGG(
      STRUCT(
        patientunitstayid,
        icu_order_within_admission,

        icu_mortality,
        hospital_mortality,

        hospitaladmitoffset,
        hospitaldischargeoffset,
        unitdischargeoffset,

        icu_admit_min_from_hospital_admit,
        icu_discharge_min_from_hospital_admit,
        hospital_discharge_min_from_hospital_admit,

        icu_los_days,
        hospital_los_days
      )
      ORDER BY icu_order_within_admission
    ) AS icu_stays,

    MIN(icu_admit_min_from_hospital_admit) AS first_icu_admit_min_from_hospital_admit,
    MAX(icu_discharge_min_from_hospital_admit) AS last_icu_discharge_min_from_hospital_admit,

    MIN(hospital_discharge_min_from_hospital_admit) AS min_hospital_discharge_min_from_hospital_admit,
    MAX(hospital_discharge_min_from_hospital_admit) AS max_hospital_discharge_min_from_hospital_admit,

    MAX(hospital_discharge_min_from_hospital_admit)
      - MIN(hospital_discharge_min_from_hospital_admit)
      AS hospital_discharge_min_range_within_admission,

    SAFE_DIVIDE(MAX(hospital_discharge_min_from_hospital_admit), 1440.0) AS hospital_los_days,

    SAFE_DIVIDE(SUM(unitdischargeoffset), 1440.0) AS total_icu_los_days,

    SAFE_DIVIDE(
      MAX(icu_discharge_min_from_hospital_admit)
        - MIN(icu_admit_min_from_hospital_admit),
      1440.0
    ) AS first_to_last_icu_span_days,

    MAX(icu_mortality) AS any_icu_mortality,
    MAX(hospital_mortality) AS hospital_mortality,

    SUM(n_diagnosis_rows) AS n_diagnosis_rows,
    SUM(n_diagnosis_labels) AS n_diagnosis_labels_row_sum,
    SUM(n_icd9codes) AS n_icd9codes_row_sum,

    SUM(CASE WHEN flag_missing_hospitaladmitoffset THEN 1 ELSE 0 END) AS n_missing_hospitaladmitoffset,
    SUM(CASE WHEN flag_missing_hospitaldischargeoffset THEN 1 ELSE 0 END) AS n_missing_hospitaldischargeoffset,
    SUM(CASE WHEN flag_missing_unitdischargeoffset THEN 1 ELSE 0 END) AS n_missing_unitdischargeoffset,

    SUM(CASE WHEN flag_icu_admit_before_hospital_admit THEN 1 ELSE 0 END) AS n_icu_admit_before_hospital_admit,
    SUM(CASE WHEN flag_icu_discharge_before_icu_admit THEN 1 ELSE 0 END) AS n_icu_discharge_before_icu_admit,
    SUM(CASE WHEN flag_nonpositive_hospital_los THEN 1 ELSE 0 END) AS n_nonpositive_hospital_los,
    SUM(CASE WHEN flag_icu_discharge_after_hospital_discharge THEN 1 ELSE 0 END) AS n_icu_discharge_after_hospital_discharge,
    SUM(CASE WHEN flag_icu_overlap_previous THEN 1 ELSE 0 END) AS n_icu_overlap_previous

  FROM unit_flags
  GROUP BY patienthealthsystemstayid
),

clean_admissions AS (
  SELECT
    *,

    (
      n_missing_hospitaladmitoffset = 0
      AND n_missing_hospitaldischargeoffset = 0
      AND n_missing_unitdischargeoffset = 0
      AND n_icu_admit_before_hospital_admit = 0
      AND n_icu_discharge_before_icu_admit = 0
      AND n_nonpositive_hospital_los = 0
      AND n_icu_discharge_after_hospital_discharge = 0
      AND n_icu_overlap_previous = 0
      AND hospital_discharge_min_range_within_admission <= 15
    ) AS is_time_clean_for_admission_trajectory

  FROM admission_agg
),

cutoff AS (
  SELECT
    APPROX_QUANTILES(first_to_last_icu_span_days, 1000)[OFFSET(995)] AS first_to_last_icu_span_days_p995
  FROM clean_admissions
  WHERE is_time_clean_for_admission_trajectory = TRUE
)

SELECT
  a.*,
  c.first_to_last_icu_span_days_p995 AS first_to_last_icu_span_days_cutoff_p995,
  TRUE AS retained_clean_admission_after_p995_trim
FROM clean_admissions a
CROSS JOIN cutoff c
WHERE a.is_time_clean_for_admission_trajectory = TRUE
  AND a.first_to_last_icu_span_days <= c.first_to_last_icu_span_days_p995;