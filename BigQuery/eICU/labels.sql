CREATE OR REPLACE TABLE `mimic-iv-484300.eICU_processed.labels` AS

SELECT
  m.uniquepid,
  m.patientunitstayid,

  b.split,

  m.gender,
  m.age,
  m.hospitaladmitoffset,
  m.hospitaldischargeoffset,
  m.unitdischargeoffset,
  m.unitstaytype,
  m.icu_mortality,
  m.hospital_mortality,

  m.diagnoses,
  m.diagnosis_labels,
  m.icd9codes

FROM `mimic-iv-484300.eICU_processed.meta` m
JOIN `mimic-iv-484300.eICU_processed.EHR_binned` b
  ON CAST(m.patientunitstayid AS STRING) = CAST(b.patientunitstayid AS STRING);