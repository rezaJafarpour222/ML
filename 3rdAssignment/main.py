import numpy as np
import pandas as pd
from solution import DecisionTree_Scenario, LDA_Scenario, SVM_Scenario
from util import splitter, z_score_scaler

raw_data = pd.read_csv("3rdAssignment/DataSets/pima-indians-diabetes.csv", header=None)
raw_data.iloc[:, 8] = raw_data.iloc[:, 8].replace(0, -1)
raw_label = raw_data.iloc[:, 8].values

raw_data = raw_data.iloc[:, 0:7].values
X_Scaled = z_score_scaler(raw_data)
weighted_input = np.hstack([np.ones((raw_data.shape[0], 1)), X_Scaled])
(X_train, Y_train, X_test, Y_test) = splitter(
    data=weighted_input, label=raw_label, splitPrecent=1.0
)
# SVM_Scenario(weighted_input=weighted_input, label=raw_label)
# LDA_Scenario(weighted_input=weighted_input, label=raw_label)
# DecisionTree_Scenario(weighted_input=weighted_input, label=raw_label)
comparison(weighted_input=weighted_input, label=raw_label)
