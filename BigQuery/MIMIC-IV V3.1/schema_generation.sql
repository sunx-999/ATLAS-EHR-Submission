-- ATLAS-EHR: Global Schema & Statistics Generation (Strict Train Split)

WITH exploded_events AS (
    -- Step 1: Flatten trajectories and filter by TRAIN split
    SELECT 
        e.normalized_concept_label,
        e.scaled_value,
        e.event_category
    FROM `mimic-iv-484300.mimic_iv_3_1_processed.ehr` t
    JOIN `mimic-iv-484300.mimic_iv_3_1_processed.labels` l ON t.hadm_id = l.hadm_id
    CROSS JOIN UNNEST(t.trajectory) AS e
    WHERE l.split = 'train'
),

feature_primary_metadata AS (
    -- Step 2: Resolve feature types from mapping
    SELECT 
        target_label,
        MAX(CASE WHEN suggested_type IN ('CONTINUOUS', 'ORDINAL') THEN 1 ELSE 0 END) AS is_numeric
    FROM `mimic-iv-484300.mimic_iv_3_1_processed.e2e_mapping`
    GROUP BY 1
),

feature_counts AS (
    -- Step 3: Global counts and Identity resolution
    SELECT 
        normalized_concept_label,
        COUNT(*) AS total_sample_count,
        COUNTIF(event_category = 'state') AS state_sample_count,
        IF(COUNTIF(event_category = 'state') > 0, 'state', 'intervention') AS feature_category
    FROM exploded_events
    GROUP BY 1
),

feature_thresholds AS (
    -- Step 4: Identify boundaries for numeric features
    SELECT 
        ee.normalized_concept_label,
        APPROX_QUANTILES(ee.scaled_value, 100)[OFFSET(5)] AS p5_threshold,
        APPROX_QUANTILES(ee.scaled_value, 100)[OFFSET(95)] AS p95_threshold
    FROM exploded_events ee
    JOIN feature_primary_metadata fpm ON ee.normalized_concept_label = fpm.target_label
    WHERE fpm.is_numeric = 1
    GROUP BY 1
),

robust_numeric_stats AS (
    -- Step 5: Calculate robust stats and include identity column
    SELECT 
        ee.normalized_concept_label,
        'NUMERIC' AS model_type,
        fc.feature_category,
        AVG(ee.scaled_value) AS mean,
        STDDEV(ee.scaled_value) AS std,
        ft.p5_threshold AS p5,
        ft.p95_threshold AS p95,
        fc.total_sample_count AS sample_count,
        fc.state_sample_count
    FROM exploded_events ee
    JOIN feature_thresholds ft ON ee.normalized_concept_label = ft.normalized_concept_label
    JOIN feature_counts fc ON ee.normalized_concept_label = fc.normalized_concept_label
    WHERE ee.scaled_value BETWEEN ft.p5_threshold AND ft.p95_threshold
    GROUP BY 1, 2, 3, ft.p5_threshold, ft.p95_threshold, fc.total_sample_count, fc.state_sample_count, fc.feature_category
),

categorical_vocab AS (
    -- Step 6: Handle Categorical tokens and include identity column
    SELECT 
        ee.normalized_concept_label,
        'CATEGORICAL' AS model_type,
        fc.feature_category,
        CAST(NULL AS FLOAT64) AS mean,
        CAST(NULL AS FLOAT64) AS std,
        CAST(NULL AS FLOAT64) AS p5,
        CAST(NULL AS FLOAT64) AS p95,
        fc.total_sample_count AS sample_count,
        fc.state_sample_count
    FROM exploded_events ee
    JOIN feature_counts fc ON ee.normalized_concept_label = fc.normalized_concept_label
    LEFT JOIN feature_primary_metadata fpm ON ee.normalized_concept_label = fpm.target_label
    LEFT JOIN robust_numeric_stats rns ON ee.normalized_concept_label = rns.normalized_concept_label
    WHERE rns.normalized_concept_label IS NULL 
      AND (
          fpm.is_numeric = 0 
          OR (fpm.target_label IS NULL AND REGEXP_CONTAINS(ee.normalized_concept_label, r'_'))
      )
    GROUP BY 1, 2, 3, fc.total_sample_count, fc.state_sample_count, fc.feature_category
)

-- Final Schema Consolidation
SELECT * FROM robust_numeric_stats
UNION ALL
SELECT * FROM categorical_vocab
ORDER BY model_type DESC, normalized_concept_label ASC;