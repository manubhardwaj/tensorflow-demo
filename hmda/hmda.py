#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools, tensorflow as tf, numpy as np, pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS=['AsofYear','RespondentID','AgencyCode','LoanType','PropertyType','LoanPurpose','Occupancy','LoanAmount000s','Preapproval','ActionType','MSAMD','StateCode','CountyCode','CensusTractNumber','ApplicantEthnicity','CoApplicantEthnicity','ApplicantRace1','ApplicantRace2','ApplicantRace3','ApplicantRace4','ApplicantRace5','CoApplicantRace1','CoApplicantRace2','CoApplicantRace3','CoApplicantRace4','CoApplicantRace5','ApplicantSex','CoApplicantSex','ApplicantIncome000s','PurchaserType','DenialReason1','DenialReason2','DenialReason3','RateSpread','HOEPAStatus','LienStatus','EditStatus','SequenceNumber','Population','MinorityPopulationPct','FFIECMedianFamilyIncome','TracttoMSAMDIncomePct','NumberofOwner-occupiedunits','Numberof1-to4-Familyunits','ApplicationDateIndicator']

FEATURES=['AsofYear','AgencyCode','LoanType','PropertyType','LoanPurpose','Occupancy','Preapproval','ActionType','MSAMD','StateCode','CountyCode','CensusTractNumber','ApplicantEthnicity','CoApplicantEthnicity','ApplicantRace1','CoApplicantRace1','ApplicantSex','CoApplicantSex','ApplicantIncome000s','PurchaserType','HOEPAStatus','LienStatus','SequenceNumber','Population','MinorityPopulationPct','FFIECMedianFamilyIncome','TracttoMSAMDIncomePct','ApplicationDateIndicator']

LABEL=['LoanAmount000s']

training_set = pd.read_csv("data/training_CHI.csv", names=COLUMNS)
testing_set = pd.read_csv("data/test_LAX.csv", names=COLUMNS)
prediction_set = pd.read_csv("data/prediction_NYC.csv", names=COLUMNS)

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[10, 10])

def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y = pd.Series(data_set[LABEL].astype(int)),
      num_epochs=num_epochs,
      shuffle=shuffle)

regressor.train(input_fn=get_input_fn(training_set), steps=5)

ev = regressor.evaluate(
    input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))

loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

y = regressor.predict(
    input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
# .predict() returns an iterator of dicts; convert to a list and print
# predictions
predictions = list(p["predictions"] for p in itertools.islice(y, 6))
print("Predictions: {}".format(str(predictions)))

