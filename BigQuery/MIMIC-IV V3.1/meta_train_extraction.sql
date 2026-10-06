-- ATLAS-EHR: Clinical Feature Discovery
-- Objective: Capture labels, units, AND value-distribution evidence.

WITH combined_raw AS (
    SELECT 
        'state' AS feature_type, s.source_table, s.concept_label,
        IFNULL(s.unit, 'unspecified') AS raw_unit,
        s.value, s.hadm_id
    FROM `mimic-iv-484300.mimic_iv_3_1_processed.states` s
    UNION ALL
    SELECT 
        'intervention' AS feature_type, i.source_table, i.concept_label,
        IFNULL(i.unit, 'unspecified') AS raw_unit,
        i.value, i.hadm_id
    FROM `mimic-iv-484300.mimic_iv_3_1_processed.interventions` i
),

-- Pre-calculate stats for each (type, source, label, unit)
feature_stats AS (
    SELECT 
        r.feature_type, r.source_table, r.concept_label, r.raw_unit,
        COUNT(*) AS occurrence_freq,
        COUNT(DISTINCT r.hadm_id) AS admission_count,
        
        -- Evidence 1: String vs Numeric Ratio (Crucial for Atomization)
        COUNTIF(SAFE_CAST(r.value AS FLOAT64) IS NULL AND r.value IS NOT NULL) AS string_val_count,
        COUNTIF(SAFE_CAST(r.value AS FLOAT64) IS NOT NULL) AS numeric_val_count,
        
        -- Evidence 2: Value Range (Crucial for Type Inference)
        MIN(SAFE_CAST(r.value AS FLOAT64)) AS min_val,
        MAX(SAFE_CAST(r.value AS FLOAT64)) AS max_val,
        COUNT(DISTINCT r.value) AS distinct_value_count,
        
        -- Evidence 3: Samples (For manual/LLM auditing)
        -- Using ARRAY_AGG with LIMIT to get a glimpse of the data
        ARRAY_AGG(DISTINCT r.value IGNORE NULLS LIMIT 5) AS sample_values
        
    FROM combined_raw r
    INNER JOIN `mimic-iv-484300.mimic_iv_3_1_processed.labels` l ON r.hadm_id = l.hadm_id
    WHERE l.split = 'train' AND l.is_standard_cohort = 1
    GROUP BY 1, 2, 3, 4
    HAVING occurrence_freq >= 50
)

SELECT 
    *,
    -- Automated Suggestion Logic to speed up Mapping Phase
    CASE 
        WHEN string_val_count > numeric_val_count THEN 'NOMINAL (Atomic Split Needed)'
        WHEN distinct_value_count = 2 AND min_val = 0 AND max_val = 1 THEN 'BINARY'
        WHEN raw_unit = 'dose' OR LOWER(concept_label) LIKE '%score%' THEN 'ORDINAL (Preserve Gradient)'
        ELSE 'CONTINUOUS'
    END AS suggested_type_hint
FROM feature_stats
ORDER BY feature_type, occurrence_freq DESC;