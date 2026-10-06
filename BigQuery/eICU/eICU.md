### EVENT:

**lab** (standardized labs, big support):

* patientunitstayid: ICU stay ID
* labresultoffset, labresultrevisedoffset: labresultoffset is the number of minutes from unit admit time that the lab value was drawn; labresultrevisedoffset is the number of minutes from unit admit time that the revised lab value was entered
* labname
* labresult, labresulttext
* labmeasurenamesystem, labmeasurenameinterface: unit



**customlab** (non-standardized labs, small support): 

* patientunitstayid
* labotheroffset
* labothername
* labotherresult, labothervaluetext



**infusiondrug** (weight info only):

* patientunitstayid
* infusionoffset
* patientweight



**intakeoutput** (output only):

* patientunitstayid
* intakeoutputoffset, intakeoutputentryoffset
* cellpath: tell which is intake, which is output
* celllabel
* cellvaluenumeric/cellvaluetext



**nurseassessment** (manual selection):

* patientunitstayid
* nurseassessoffset, nurseassessentryoffset: careful some measurement last an interval of time, e.g. pulse
* cellattributepath
* celllabel + cellattribute
* cellattributevalue



**nursecharting** (manual selection):

* patientunitstayid
* nursingchartoffset, nursingchartentryoffset: careful some measurement last an interval of time
* nursingchartcelltypecat: filter
* nursingchartcelltypevallabel + nursingchartcelltypevalname
* nursingchartvalue



**physicalexam** (manual selection):

* patientunitstayid
* physicalexamoffset
* physicalexampath: filter
* physicalexamvalue, physicalexamtext



**respiratorycharting** (manual selection):

* patientunitstayid
* respchartoffset, respchartentryoffset
* respcharttypecat: filter
* respchartvaluelabel
* respchartvalue



**vitalaperiodic**:

* patientunitstayid
* observationoffset
* noninvasivesystolic/noninvasivediastolic/noninvasivemean/paop/cardiacoutput/cardiacinput/svr/svri/pvr/pvri



**vitalperiodic**:

* patientunitstayid
* observationoffset
* temperature/sao2/heartrate/respiration/cvp/etco2/systemicsystolic/systemicdiastolic/systemicmean/pasystolic/padiastolic/pamean/st1/st2/st3/icp





### META:

**patient** (only year and clock time available, no date):

* uniquepid: split
* patienthealthsystemstayid: hospital admission
* patientunitstayuid: icu admission
* gender/age
* hospitaladmitoffset: hospital admission time relative to ICU admission time
* hospitaldischargeoffset: hospital discharge time relative to ICU admission time
* unitdischargeoffset: ICU discharge time relative to ICU admission time
* unitdischargestatus: expired=death
* hospitaldischargestatus: expired=death
* unitstaytype: admit/stepdown/transfer/readmit





### LABEL:

**diagnosis**:

* patientunitstayid
* diagnosisstring: differential diagnosis
* diagnosisoffset
* icd9code





### Potential meta data:

**admissiondrug**: medications that a patient was taking prior to admission to the ICU. Medication history is typically entered into the system when compiling admission notes, progress notes, or medication reconciliation records after ICU admission; consequently, the time of entry often falls after the patient has been admitted to the ICU.

**admissiondx**: why admitted to ICU

allergy: when allergy info become accessible

**pasthistory**: patient’s history, including comorbidities, risk factors, therapies, health care providers, recent hospitalizations

**patient**: hospitaladmitsource, unitadmitsource, unittype





### Potential benchmark (cutoff first 24h):

**apacheapsvar**: worst acute physiology summary on the APACHE day (usually first 24h admitted to ICU)

**apachepredvar**: full input variables to APACHE prediction model: demographics/meta + APACHE admission diagnosis/surgery history + chronic disease + GCS/drugs + ventilation/breathing/ABG + others

**apachepatientresult**: APACHE final prediction results, two rows per patientUnitStayID: one for APACHE IV, one for APACHE IVa



### Ruled-out:

**careplanprovider**

**careplaneol**: end-of-life discussion

**careplangeneral**

**careplangoal**

**careplaninfectiousdisease**: record of infectious diseases added as part of the care plan form

**hospital**

**infusiondrug**

**medication**

**microlab**: delayed result

**note**

**nursecare**

**respiratorycare**

**treatment**

