-- ATLAS-EHR Audit: Part 1 - Cohort & Splits
SELECT 
    split,
    -- Population counts
    COUNT(DISTINCT subject_id) AS n_subjects,
    COUNT(DISTINCT hadm_id) AS n_admissions,
    -- Outcome distributions
    ROUND(AVG(label_mortality), 4) AS mortality_rate,
    ROUND(AVG(label_readmission_30d), 4) AS readmission_rate,
    -- LOS distributions
    MIN(label_los_days) AS los_min,
    MAX(label_los_days) AS los_max,
    ROUND(AVG(label_los_days), 2) AS los_mean,
    APPROX_QUANTILES(label_los_days, 100)[OFFSET(50)] AS los_median,
    -- Exclusion metrics
    SUM(flag_is_pediatric) AS count_pediatric,
    SUM(flag_is_logical_error) AS count_logical_error,
    SUM(flag_is_short_stay) AS count_short_stay,
    SUM(CASE WHEN flag_is_pediatric = 0 AND flag_is_logical_error = 0 THEN 1 ELSE 0 END) AS n_standard_cohort
FROM `mimic-iv-484300.mimic_iv_3_1_processed.labels`
GROUP BY split
ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'val' THEN 2 ELSE 3 END;

-- ATLAS-EHR Audit: Part 2 - State Vocabulary (Filtered by Standard Cohort)
SELECT 
    s.source_table,
    -- Items with hard identifiers (ITEMIDs)
    COUNT(DISTINCT s.concept_id) AS structured_ids,
    -- Items with only textual descriptors
    COUNT(DISTINCT s.concept_label) AS text_labels,
    -- Total logical entities to be embedded by Bio_ClinicalBERT
    COUNT(DISTINCT COALESCE(s.concept_id, s.concept_label)) AS total_unique_entities
FROM `mimic-iv-484300.mimic_iv_3_1_processed.states` s
INNER JOIN `mimic-iv-484300.mimic_iv_3_1_processed.labels` l 
    ON s.hadm_id = l.hadm_id
WHERE l.flag_is_pediatric = 0 AND l.flag_is_logical_error = 0
GROUP BY s.source_table
ORDER BY total_unique_entities DESC;

-- ATLAS-EHR Audit: Part 3 - Intervention Vocabulary (Filtered by Standard Cohort)
SELECT 
    i.source_table,
    -- ICU-based structured actions
    COUNT(DISTINCT i.concept_id) AS structured_ids,
    -- Ward-based textual medications (eMAR)
    COUNT(DISTINCT i.concept_label) AS text_labels,
    -- Unified action manifold size
    COUNT(DISTINCT COALESCE(i.concept_id, i.concept_label)) AS total_unique_entities
FROM `mimic-iv-484300.mimic_iv_3_1_processed.interventions` i
INNER JOIN `mimic-iv-484300.mimic_iv_3_1_processed.labels` l 
    ON i.hadm_id = l.hadm_id
WHERE l.flag_is_pediatric = 0 AND l.flag_is_logical_error = 0
GROUP BY i.source_table
ORDER BY total_unique_entities DESC;