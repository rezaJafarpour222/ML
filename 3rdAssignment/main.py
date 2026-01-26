import numpy as np
import pandas as pd

from util import splitter, z_score_scaler
import comparison

raw_data = pd.read_csv("Perceptron/DataSets/pima-indians-diabetes.csv", header=None)
raw_data.iloc[:, 8] = raw_data.iloc[:, 8].replace(0, -1)
raw_label = raw_data.iloc[:, 8].values

raw_data = raw_data.iloc[:, 0:7].values
X_Scaled = z_score_scaler(raw_data)
weighted_input = np.hstack([np.ones((raw_data.shape[0], 1)), X_Scaled])
(X_train, Y_train, X_test, Y_test) = splitter(
    data=weighted_input, label=raw_label, splitPrecent=1.0
)
comparison.SVM_metrics(weightedInput=weighted_input, label=raw_label)
comparison.LDA_metrics(weightedInput=weighted_input, label=raw_label)
comparison.DecisionTree_metrics(weightedInput=weighted_input, label=raw_label)
comparison.models_comparison(weightedInput=weighted_input, label=raw_label)
