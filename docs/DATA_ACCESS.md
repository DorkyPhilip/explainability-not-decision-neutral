# Obtaining the source datasets

Neither source dataset is redistributed in this repository. Both are freely
available to credentialed researchers from PhysioNet, and both require
completion of a human-subjects research training course and acceptance of a data
use agreement.

## Primary cohort: MC-MED

Multimodal Clinical Monitoring in the Emergency Department, version 1.0.1.

- DOI: [10.13026/wvyw-g663](https://doi.org/10.13026/wvyw-g663)
- Access: PhysioNet credentialed access, data use agreement required
- Content: de-identified adult emergency department visits from an academic
  medical center, with triage vital signs, chief complaint, prior diagnoses,
  active medications, and the treating clinician's disposition

## Replication cohort: MIMIC-IV-ED

Version 2.2.

- DOI: [10.13026/5ntk-km72](https://doi.org/10.13026/5ntk-km72)
- Access: PhysioNet credentialed access, data use agreement required
- Content: approximately 425,000 de-identified emergency department stays from a
  different academic medical center

## Locating the study encounters

`data/cohort_manifest.csv` lists, for each study encounter, the identifier it
carries in the source dataset: the `CSN` column of the MC-MED `visits` table, or
the `stay_id` column of the MIMIC-IV-ED `edstays` table. With credentialed access
you can select the 2000 encounters of each cohort directly from your own copy,
without running the cohort builders. The manifest contains no other field.

## Expected local schema

The cohort builders read from a DuckDB database per dataset. `--db` points at
that file. The builders expect these relations:

**MC-MED**

| Relation | Columns used |
| --- | --- |
| `v_triage_capsule_raw` | `CSN`, `Age`, `Gender`, `Means_of_arrival`, `Triage_acuity`, `Triage_HR`, `Triage_SBP`, `Triage_DBP`, `Triage_RR`, `Triage_SpO2`, `Triage_Temp`, `Triage_Pain`, `CC` |
| `visits` | `CSN`, `MRN`, `ED_dispo`, `Race`, `Payor_class` |
| `pmh` | `MRN`, `DescCCS` |
| `meds` | `MRN`, `Generic_name`, `Active` |

**MIMIC-IV-ED**

| Relation | Columns used |
| --- | --- |
| `v_mimic_capsule` | `stay_id`, `age`, `gender`, `arrival_transport`, `triage_acuity`, `triage_hr`, `triage_sbp`, `triage_dbp`, `triage_rr`, `triage_spo2`, `triage_temp`, `triage_pain`, `cc`, `disposition`, `race`, `med_list` |
| `v_pmh_prior` | `stay_id`, `pmh_list` |

`v_triage_capsule_raw`, `v_mimic_capsule`, and `v_pmh_prior` are thin views over
the distributed tables: they join the triage record to the encounter and
aggregate prior diagnoses and active medications per patient. `v_pmh_prior`
restricts prior diagnoses to those recorded before the index encounter, so no
information from the index visit leaks into the prompt.

## What the model was shown

The model never saw the source tables. It saw only the rendered capsule
produced by `cohort/capsule.py`, which is a fixed six-line summary:

```
PATIENT: 54yo F via EMS | ESI-3
VITALS: HR 96 | BP 138/82 | RR 18 | SpO2 97% | Temp 37.1C | Pain 6/10
CC: ABDOMINAL PAIN
FLAGS: none
PMH: Diabetes mellitus without complication; Essential hypertension
MEDICATIONS: lisinopril; metformin
```

No free-text notes, laboratory results, imaging, or disposition-adjacent fields
were included, and no identifiers were present in the rendered text.
