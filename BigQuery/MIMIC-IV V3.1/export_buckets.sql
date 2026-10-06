-- ATLAS-EHR: Manual Partitioned Export to GCS
-- Logic: Iterates through each split and bucket_id to create a highly parallelizable directory structure.

BEGIN
  DECLARE splits ARRAY<STRING> DEFAULT ['train', 'val', 'test'];
  DECLARE b_id INT64 DEFAULT 0;

  FOR split_item IN (SELECT * FROM UNNEST(splits) AS s) DO
    
    WHILE b_id < 100 DO
      
      EXECUTE IMMEDIATE FORMAT("""
        EXPORT DATA OPTIONS(
          uri='gs://bucket-20260222/mimic_iv/%s/bucket=%d/shard-*.jsonl.gz',
          format='JSON',
          compression='GZIP',
          overwrite=true
        ) AS
        SELECT * EXCEPT(split, bucket_id)
        FROM `mimic-iv-484300.mimic_iv_3_1_processed.ehr_model_ready`
        WHERE split = '%s' AND bucket_id = %d
      """, split_item.s, b_id, split_item.s, b_id);
      
      SET b_id = b_id + 1;
    END WHILE;
    
    SET b_id = 0; 
  END FOR;
END;