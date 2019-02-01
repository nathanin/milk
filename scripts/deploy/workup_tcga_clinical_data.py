import pandas as pd
import numpy as np
import pickle

fields = [
  'patient.bcr_patient_barcode',
  'patient.bcr_patient_uuid', 
  'patient.days_to_first_biochemical_recurrence', 
  'patient.days_to_last_followup', 
  'patient.follow_ups.follow_up.days_to_death', 
  'patient.follow_ups.follow_up.days_to_first_biochemical_recurrence', 
  'patient.follow_ups.follow_up.days_to_last_followup', 
  'patient.follow_ups.follow_up.days_to_new_tumor_event_after_initial_treatment', 
  'patient.number_of_lymphnodes_positive_by_he', 
  'patient.person_neoplasm_cancer_status', 
  'patient.primary_therapy_outcome_success', 
  'patient.stage_event.gleason_grading.gleason_score', 
  'patient.stage_event.gleason_grading.primary_pattern', 
  'patient.stage_event.gleason_grading.secondary_pattern', 
  'patient.stage_event.psa.days_to_psa', 
  'patient.stage_event.psa.psa_value', 
]

df = pd.read_csv('gdac.broadinstitute.org_PRAD.Merge_Clinical.Level_1.2016012800.0.0/PRAD.clin.merged.txt', sep='\t', index_col=None, header=None)
print(df.head())
print(df.shape)

use_rows = np.array([x in fields for x in df[0].values])
df = df.loc[use_rows, :]
# df.reset_index(inplace=True, drop=True)

print(df.head())
print(df.shape)

df.to_csv('prad_clinical_data.csv')

""" Process time-to data

  - biochemical recurrence
  - new tumor event
  - death -- not very useful

"""
first_bio_recurr = {} ## tuples: (time-to, censored)
for col in df.columns.values[1:]:
  barcode = df.loc[11, col].upper()
  bcr = df.loc[21, col]
  last_fup = df.loc[183, col]
  try:
    last_fup = float(last_fup)
    if np.isnan(last_fup):
      continue
  except:
    last_fup = np.inf

  print(col, barcode, bcr, last_fup)

  # For lifelines, we need (time-to, "event")
  #  where time-to is a float
  #  and "event" is a bool indicating whether the event was observed (1) or the individual was censored (0)
  try:
    bcr = float(bcr)
    if bcr < last_fup and not np.isnan(bcr):
      first_bio_recurr[barcode] = (bcr, 1)
    else:
      first_bio_recurr[barcode] = (last_fup, 0)
  except:
    continue

pickle.dump(first_bio_recurr, open('prad_days_to_biochemical_recurrence.pkl', 'bw+'))

first_new_tumor_event = {}
for col in df.columns.values[1:]:
  barcode = df.loc[11, col].upper()
  nte = df.loc[184, col]
  last_fup = df.loc[183, col]

  try:
    last_fup = float(last_fup)
    if np.isnan(last_fup):
      continue
  except:
    last_fup = np.inf

  print(col, barcode, nte, last_fup)

  try:
    nte = float(nte)
    if nte < last_fup and not np.isnan(nte):
      first_new_tumor_event[barcode] = (nte, 1)
    else:
      first_new_tumor_event[barcode] = (last_fup, 0)
  except:
    continue

pickle.dump(first_new_tumor_event, open('prad_days_to_new_tumor_event.pkl', 'bw+'))

""" Process categorical data

  - positive lymphnodes
  - neoplasm status
  - gleason score
"""
