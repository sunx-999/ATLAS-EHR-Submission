-- eICU: Manual bucketed export to GCS
-- Input table:
--   mimic-iv-484300.eICU_processed.EHR_binned
--
-- EHR_binned schema:
--   patientunitstayid
--   uniquepid
--   admission_id
--   split
--   bucket_id
--   trajectory ARRAY<STRUCT<
--     timestamp INT64,
--     mimic_vid INT64,
--     converted_value FLOAT64
--   >>
--   n_events
--   n_unique_timestamps
--   first_timestamp
--   last_timestamp
--
-- Export layout:
--   gs://eicu-20260629/train/bucket=0/shard-*.jsonl.gz
--   gs://eicu-20260629/train/bucket=1/shard-*.jsonl.gz
--   ...
--   gs://eicu-20260629/val/bucket=0/shard-*.jsonl.gz
--   ...
--   gs://eicu-20260629/test/bucket=99/shard-*.jsonl.gz
--
-- Notes:
--   - split and bucket_id are used only for export partitioning.
--   - They are excluded from the exported records.
--   - Each exported row is one ICU stay trajectory.

BEGIN
  DECLARE splits ARRAY<STRING> DEFAULT ['train', 'val', 'test'];
  DECLARE b_id INT64 DEFAULT 0;

  FOR split_item IN (SELECT * FROM UNNEST(splits) AS s) DO

    WHILE b_id < 100 DO

      EXECUTE IMMEDIATE FORMAT("""
        EXPORT DATA OPTIONS(
          uri='gs://eicu-20260629/%s/bucket=%d/shard-*.jsonl.gz',
          format='JSON',
          compression='GZIP',
          overwrite=true
        ) AS
        SELECT
          patientunitstayid,
          uniquepid,
          admission_id,
          trajectory,
          n_events,
          n_unique_timestamps,
          first_timestamp,
          last_timestamp
        FROM `mimic-iv-484300.eICU_processed.EHR_binned`
        WHERE split = '%s'
          AND bucket_id = %d
      """, split_item.s, b_id, split_item.s, b_id);

      SET b_id = b_id + 1;
    END WHILE;

    SET b_id = 0;
  END FOR;
END;