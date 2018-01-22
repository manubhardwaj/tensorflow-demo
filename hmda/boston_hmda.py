#!/usr/bin/python
#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""DNNRegressor with custom input_fn for Housing dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS=['AsofYear','RespondentID','AgencyCode','LoanType','PropertyType','LoanPurpose','Occupancy','LoanAmount000s','Preapproval','ActionType','MSAMD','StateCode','CountyCode','CensusTractNumber','ApplicantEthnicity','CoApplicantEthnicity','ApplicantRace1','ApplicantRace2','ApplicantRace3','ApplicantRace4','ApplicantRace5','CoApplicantRace1','CoApplicantRace2','CoApplicantRace3','CoApplicantRace4','CoApplicantRace5','ApplicantSex','CoApplicantSex','ApplicantIncome000s','PurchaserType','DenialReason1','DenialReason2','DenialReason3','RateSpread','HOEPAStatus','LienStatus','EditStatus','SequenceNumber','Population','MinorityPopulationPct','FFIECMedianFamilyIncome','TracttoMSAMDIncomePct','NumberofOwner-occupiedunits','Numberof1-to4-Familyunits','ApplicationDateIndicator']

FEATURES=['ApplicantIncome000s','Population']
#FEATURES=['LoanType','PropertyType','Preapproval','CensusTractNumber','ApplicantEthnicity','ApplicantRace1','ApplicantSex','ApplicantIncome000s','PurchaserType','DenialReason1','Population','MinorityPopulationPct','FFIECMedianFamilyIncome']

#FEATURES=['AsofYear','AgencyCode','LoanType','PropertyType','LoanPurpose','Occupancy','Preapproval','ActionType','MSAMD','StateCode','CountyCode','CensusTractNumber','ApplicantEthnicity','CoApplicantEthnicity','ApplicantRace1','ApplicantRace2','ApplicantRace3','ApplicantRace4','ApplicantRace5','CoApplicantRace1','CoApplicantRace2','CoApplicantRace3','CoApplicantRace4','CoApplicantRace5','ApplicantSex','CoApplicantSex','ApplicantIncome000s','PurchaserType','DenialReason1','DenialReason2','DenialReason3','RateSpread','HOEPAStatus','LienStatus','EditStatus','SequenceNumber','Population','MinorityPopulationPct','FFIECMedianFamilyIncome','TracttoMSAMDIncomePct','NumberofOwner-occupiedunits','Numberof1-to4-Familyunits','ApplicationDateIndicator']

LABEL='LoanAmount000s'

def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)

def main(unused_argv):
  # Load datasets
  training_set = pd.read_csv("data/training.csv", names=COLUMNS)
  test_set = pd.read_csv("data/test.csv", names=COLUMNS)

  prediction_set = pd.read_csv("data/prediction.csv", names=COLUMNS)

  for i in list(training_set):
      if(i != 'RespondentID'):
          training_set[i] = pd.to_numeric(training_set[i],errors='coerce')

  #training_set.apply(pd.to_numeric)
  #test_set.apply(pd.to_numeric)
  #prediction_set.apply(pd.to_numeric)

#
#  pd.to_numeric(training_set['ApplicantIncome000s'], errors='coerce')
#  pd.to_numeric(test_set['ApplicantIncome000s'], errors='coerce')
#  pd.to_numeric(prediction_set['ApplicantIncome000s'], errors='coerce')
#
#  pd.to_numeric(training_set['Population'], errors='coerce')
#  pd.to_numeric(test_set['Population'], errors='coerce')
#  pd.to_numeric(prediction_set['Population'], errors='coerce')
#
#  pd.to_numeric(training_set['TracttoMSAMDIncomePct'], errors='coerce')
#  pd.to_numeric(test_set['TracttoMSAMDIncomePct'], errors='coerce')
#  pd.to_numeric(prediction_set['TracttoMSAMDIncomePct'], errors='coerce')
#

  # Feature cols
  feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                        hidden_units=[5, 5],
                                        optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=1e-2,l1_regularization_strength=0.001))

  # Train
  regressor.train(input_fn=get_input_fn(training_set), steps=50)

  # Evaluate loss over one epoch of test_set.
  ev = regressor.evaluate(
      input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

  # Print out predictions over a slice of prediction_set.
  y = regressor.predict(
      input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
  # .predict() returns an iterator of dicts; convert to a list and print
  # predictions
  predictions = list(p["predictions"] for p in itertools.islice(y, 6))
  print("Predictions: {}".format(str(predictions)))

if __name__ == "__main__":
  tf.app.run()
