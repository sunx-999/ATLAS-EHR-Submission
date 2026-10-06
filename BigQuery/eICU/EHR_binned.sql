-- eICU: Group by ICU stay, sort by timestamp, pack model-ready events into trajectory bins.
-- Input:
--   mimic-iv-484300.eICU_processed.EHR_model_ready
--   mimic-iv-484300.eICU_processed.labels
--
-- Output:
--   mimic-iv-484300.eICU_processed.EHR_binned
--
-- Notes:
--   - EHR_model_ready already contains mimic_vid and converted_value.
--   - source / label / value / eicu_type / mimic_type / normalized_concept_label are not retained.
--   - event_order_within_timestamp is used only for stable ordering, then dropped.
--   - split is taken from labels, not recomputed, so it stays exactly consistent with labels.

CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.EHR_binned`
CLUSTER BY patientunitstayid AS

WITH labels_one AS (
  SELECT
    CAST(patientunitstayid AS INT64) AS patientunitstayid,
    ANY_VALUE(split) AS split
  FROM `mimic-iv-484300.eICU_processed.labels`
  GROUP BY patientunitstayid
),

packed AS (
  SELECT
    CAST(e.patientunitstayid AS INT64) AS patientunitstayid,
    ANY_VALUE(e.uniquepid) AS uniquepid,
    ANY_VALUE(e.admission_id) AS admission_id,

    ARRAY_AGG(
      STRUCT(
        CAST(e.timestamp AS INT64) AS timestamp,
        CAST(e.mimic_vid AS INT64) AS mimic_vid,
        CAST(e.converted_value AS FLOAT64) AS converted_value
      )
      ORDER BY
        CAST(e.timestamp AS INT64) ASC,
        CAST(e.event_order_within_timestamp AS INT64) ASC,
        CAST(e.mimic_vid AS INT64) ASC
    ) AS trajectory,

    COUNT(*) AS n_events,
    COUNT(DISTINCT e.timestamp) AS n_unique_timestamps,
    MIN(CAST(e.timestamp AS INT64)) AS first_timestamp,
    MAX(CAST(e.timestamp AS INT64)) AS last_timestamp

  FROM `mimic-iv-484300.eICU_processed.EHR_model_ready` e
  GROUP BY CAST(e.patientunitstayid AS INT64)
)

SELECT
  p.patientunitstayid,
  p.uniquepid,
  p.admission_id,

  l.split,
  MOD(ABS(FARM_FINGERPRINT(CAST(p.uniquepid AS STRING))), 100) AS bucket_id,

  p.trajectory,

  p.n_events,
  p.n_unique_timestamps,
  p.first_timestamp,
  p.last_timestamp

FROM packed p
JOIN labels_one l
  ON p.patientunitstayid = l.patientunitstayid;