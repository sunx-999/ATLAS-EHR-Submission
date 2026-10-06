-- Runtime BigQuery diagnostics for the eICU external-validation audit.
-- Replace `mimic-iv-484300.eICU_processed` only if the processed dataset differs.

-- 1. Cohort and split counts / split multiplicity.
SELECT split, COUNT(*) AS n_rows, COUNT(DISTINCT patientunitstayid) AS n_stays
FROM `mimic-iv-484300.eICU_processed.labels`
GROUP BY split ORDER BY split;

SELECT patientunitstayid, COUNT(DISTINCT split) AS n_splits, ARRAY_AGG(DISTINCT split ORDER BY split) AS splits
FROM `mimic-iv-484300.eICU_processed.labels`
GROUP BY patientunitstayid HAVING n_splits > 1;

-- 2. LOS units and task prevalence.  Use the actual label artifact columns.
SELECT
  COUNT(*) AS n,
  MIN(unit_los) AS min_unit_los,
  APPROX_QUANTILES(unit_los, 100)[OFFSET(50)] AS median_unit_los,
  APPROX_QUANTILES(unit_los, 100)[OFFSET(95)] AS p95_unit_los,
  AVG(CAST(unit_los >= 3 AS INT64)) AS los_ge_3d_rate,
  AVG(CAST(unit_los >= 7 AS INT64)) AS los_ge_7d_rate
FROM `mimic-iv-484300.eICU_processed.labels`;

SELECT split, AVG(CAST(icu_mortality=1 AS INT64)) AS mortality_rate
FROM `mimic-iv-484300.eICU_processed.labels` GROUP BY split ORDER BY split;

-- 3. Timeline range and negative timestamps.
SELECT
  COUNT(*) AS n_events,
  COUNTIF(timestamp < 0) AS n_negative,
  MIN(timestamp) AS min_timestamp,
  MAX(timestamp) AS max_timestamp
FROM `mimic-iv-484300.eICU_processed.EHR_model_ready`;

-- 4. Duplicate events before packing.
SELECT patientunitstayid, timestamp, mimic_vid, COUNT(*) AS n
FROM `mimic-iv-484300.eICU_processed.EHR_model_ready`
GROUP BY patientunitstayid, timestamp, mimic_vid HAVING n > 1
ORDER BY n DESC LIMIT 1000;

-- 5. Mapping/source coverage and VID coverage.
SELECT source, COUNT(*) AS n_events, COUNT(DISTINCT patientunitstayid) AS n_stays,
       COUNT(DISTINCT mimic_vid) AS n_vids
FROM `mimic-iv-484300.eICU_processed.EHR_model_ready`
GROUP BY source ORDER BY n_events DESC;

SELECT mimic_vid, COUNT(*) AS n_events, COUNT(DISTINCT patientunitstayid) AS n_stays
FROM `mimic-iv-484300.eICU_processed.EHR_model_ready`
GROUP BY mimic_vid ORDER BY n_events DESC;

-- 6. Per-stay sequence size and span.
SELECT patientunitstayid, COUNT(*) AS n_events, COUNT(DISTINCT timestamp) AS n_timestamps,
       MIN(timestamp) AS first_timestamp, MAX(timestamp) AS last_timestamp
FROM `mimic-iv-484300.eICU_processed.EHR_model_ready`
GROUP BY patientunitstayid;

-- 7. Multiple ICU stays in one hospital stay.
SELECT patienthealthsystemstayid, COUNT(DISTINCT patientunitstayid) AS n_icu_stays
FROM `mimic-iv-484300.eICU_processed.patient_base`
GROUP BY patienthealthsystemstayid HAVING n_icu_stays > 1;

-- 8. Numeric cast / normalization failure diagnostics before final filtering.
SELECT e.source, e.label,
       COUNT(*) AS n,
       COUNTIF(SAFE_CAST(REGEXP_EXTRACT(CAST(e.value AS STRING), r'^[<>~= ]*([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))') AS FLOAT64) IS NULL) AS n_numeric_cast_fail
FROM `mimic-iv-484300.eICU_processed.EHR_trimmed_outlier_removed` e
GROUP BY e.source, e.label ORDER BY n DESC;

-- 9. Unit fallback rates (source-specific columns are queried directly).
SELECT COUNT(*) AS n, COUNTIF(labresultoffset IS NULL AND labresultrevisedoffset IS NOT NULL) AS n_revised_fallback
FROM `physionet-data.eicu_crd.lab`;
SELECT COUNT(*) AS n, COUNTIF(intakeoutputoffset IS NULL AND intakeoutputentryoffset IS NOT NULL) AS n_entry_fallback
FROM `physionet-data.eicu_crd.intakeoutput`;
