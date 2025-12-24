import numpy as np
import pandas as pd
from solutions import (
    effect_of_learning_rate_scenario,
    effect_of_sample_size_scenario,
    models_comparison_data_percentage_scenario,
    models_comparison_learning_rate_scenario,
    precision_accuracy_curves_scenario,
)
from utils import plotter, splitter, z_score_scaler

raw_data = pd.read_csv("Perceptron/DataSets/pima-indians-diabetes.csv", header=None)
raw_data.iloc[:, 8] = raw_data.iloc[:, 8].replace(0, -1)
raw_label = raw_data.iloc[:, 8].values

raw_data = raw_data.iloc[:, 0:7].values
X_Scaled = z_score_scaler(raw_data)
weighted_input = np.hstack([np.ones((raw_data.shape[0], 1)), X_Scaled])
(X_train, Y_train, X_test, Y_test) = splitter(
    data=weighted_input, label=raw_label, splitPrecent=1.0
)

precision_accuracy_curves_scenario(weightedInput=weighted_input, label=raw_label)
# effect_of_sample_size_scenario(weightedInput=weighted_input, label=raw_label)
# effect_of_learning_rate_scenario(weightedInput=weighted_input, label=raw_label)
# models_comparison_data_percentage_scenario(
# weightedInput=weighted_input, label=raw_label
# )
# models_comparison_learning_rate_scenario(weightedInput=weighted_input, label=raw_label)
