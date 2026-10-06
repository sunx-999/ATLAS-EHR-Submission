-- Objective: Generate clean, unit-aware trajectories for clinical pre-training.
-- Refinements: 98th Percentile Pruning, Unit-Aware Joins, Source Context Isolation.

-- STEP 1: Global Frequency Filter (The "Noise Gate")
-- Ensures only robust clinical signals (n >= 50) are modeled.
WITH global_concept_counts AS (
    SELECT 
        concept_label, source_table, COUNT(*) as freq
    FROM (
        SELECT concept_label, source_table FROM `mimic-iv-484300.mimic_iv_3_1_processed.states`
        UNION ALL
        SELECT concept_label, source_table FROM `mimic-iv-484300.mimic_iv_3_1_processed.interventions`
    )
    GROUP BY 1, 2
    HAVING freq >= 50 
),

-- STEP 2: Global Backbone for trajectory alignment
-- Creates a distinct timeline for each admission.
global_time_backbone AS (
    SELECT DISTINCT subject_id, hadm_id, timestamp
    FROM (
        SELECT s.subject_id, s.hadm_id, s.charttime AS timestamp
        FROM `mimic-iv-484300.mimic_iv_3_1_processed.states` s
        INNER JOIN global_concept_counts gcc ON s.concept_label = gcc.concept_label AND s.source_table = gcc.source_table
        UNION DISTINCT
        SELECT i.subject_id, i.hadm_id, i.starttime AS timestamp
        FROM `mimic-iv-484300.mimic_iv_3_1_processed.interventions` i
        INNER JOIN global_concept_counts gcc ON i.concept_label = gcc.concept_label AND i.source_table = gcc.source_table
        UNION DISTINCT
        SELECT i.subject_id, i.hadm_id, i.endtime AS timestamp
        FROM `mimic-iv-484300.mimic_iv_3_1_processed.interventions` i
        INNER JOIN global_concept_counts gcc ON i.concept_label = gcc.concept_label AND i.source_table = gcc.source_table
    )
),

-- STEP 3: Processed States (Executing Atomic logic + Unit-Aware Semantic Splitting)
processed_states AS (
    SELECT 
        s.subject_id, s.hadm_id, s.charttime AS timestamp,
        -- LOGIC: Atomize Categorical values or use the Split Target Label
        CASE 
            WHEN IFNULL(m.suggested_type, 'CONTINUOUS') = 'NOMINAL' OR SAFE_CAST(CAST(s.value AS STRING) AS FLOAT64) IS NULL 
                THEN CONCAT(COALESCE(m.target_label, s.concept_label), '_', LOWER(REGEXP_REPLACE(TRIM(CAST(s.value AS STRING)), r'[^a-zA-Z0-9]', '')))
            ELSE COALESCE(m.target_label, s.concept_label)
        END AS normalized_concept_label,
        
        -- LOGIC: Unit Normalization with Multiplier/Offset
        CASE 
            WHEN IFNULL(m.suggested_type, 'NOMINAL') = 'NOMINAL' OR SAFE_CAST(CAST(s.value AS STRING) AS FLOAT64) IS NULL THEN 1.0
            ELSE (SAFE_CAST(CAST(s.value AS STRING) AS FLOAT64) + IFNULL(m.offset, 0.0)) * IFNULL(m.multiplier, 1.0)
        END AS scaled_value,
        
        COALESCE(m.target_unit, s.unit) AS target_unit,
        'state' AS event_category, s.source_table
    FROM `mimic-iv-484300.mimic_iv_3_1_processed.states` s
    INNER JOIN global_concept_counts gcc ON s.concept_label = gcc.concept_label AND s.source_table = gcc.source_table
    -- ROOT FIX: Join on Unit + Source + Label to prevent Cartesian explosion
    LEFT JOIN `mimic-iv-484300.mimic_iv_3_1_processed.e2e_mapping` m 
      ON s.concept_label = m.raw_label 
      AND IFNULL(s.unit, 'unspecified') = m.raw_unit 
      AND s.source_table = m.source_table
    WHERE s.value IS NOT NULL
),

-- STEP 4: Processed Interventions
processed_interventions AS (
    SELECT 
        i.subject_id, i.hadm_id, gtb.timestamp,
        CASE 
            WHEN IFNULL(m.suggested_type, 'CONTINUOUS') = 'NOMINAL' OR SAFE_CAST(CAST(i.value AS STRING) AS FLOAT64) IS NULL 
                THEN CONCAT(COALESCE(m.target_label, i.concept_label), '_', LOWER(REGEXP_REPLACE(TRIM(CAST(i.value AS STRING)), r'[^a-zA-Z0-9]', '')))
            ELSE COALESCE(m.target_label, i.concept_label)
        END AS normalized_concept_label,
        
        CASE 
            WHEN IFNULL(m.suggested_type, 'NOMINAL') = 'NOMINAL' OR SAFE_CAST(CAST(i.value AS STRING) AS FLOAT64) IS NULL THEN 1.0
            ELSE (SAFE_CAST(CAST(i.value AS STRING) AS FLOAT64) + IFNULL(m.offset, 0.0)) * IFNULL(m.multiplier, 1.0)
        END AS scaled_value,
        
        COALESCE(m.target_unit, i.unit) AS target_unit,
        'intervention' AS event_category, i.source_table
    FROM global_time_backbone gtb
    INNER JOIN `mimic-iv-484300.mimic_iv_3_1_processed.interventions` i ON gtb.hadm_id = i.hadm_id
    INNER JOIN global_concept_counts gcc ON i.concept_label = gcc.concept_label AND i.source_table = gcc.source_table
    -- ROOT FIX: Sync Join logic for interventions
    LEFT JOIN `mimic-iv-484300.mimic_iv_3_1_processed.e2e_mapping` m
        ON i.concept_label = m.raw_label 
        AND IFNULL(i.unit, 'unspecified') = m.raw_unit 
        AND i.source_table = m.source_table
    WHERE i.value IS NOT NULL 
      AND i.starttime != i.endtime
      AND gtb.timestamp BETWEEN i.starttime AND i.endtime
),

-- STEP 5: Aggregation into Sequences
final_aggregated AS (
    SELECT 
        subject_id, hadm_id,
        ARRAY_AGG(
            STRUCT(
                timestamp,
                normalized_concept_label,
                scaled_value,
                target_unit,
                event_category,
                source_table
            ) ORDER BY timestamp ASC
        ) AS trajectory,
        COUNT(DISTINCT timestamp) AS num_timestamps
    FROM (
        SELECT * FROM processed_states
        UNION ALL
        SELECT * FROM processed_interventions
    )
    GROUP BY subject_id, hadm_id
),

-- STEP 6: Final Outlier Removal based on 98th percentile
-- Filters trajectories longer than P98 to prevent OOM in Transformer
length_thresholds AS (
    SELECT 
        APPROX_QUANTILES(num_timestamps, 100)[OFFSET(98)] AS p98_ts_limit
    FROM final_aggregated
)

SELECT 
    t.*,
    lt.p98_ts_limit
FROM final_aggregated t, length_thresholds lt
WHERE t.num_timestamps > 1             -- Filter out non-sequential data
  AND t.num_timestamps <= lt.p98_ts_limit;