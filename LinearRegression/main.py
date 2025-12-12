from solutions import (
    effect_of_learning_rate_scenario,
    effect_of_sample_size_scenario,
    model_comparison_prediction_scenario,
    models_comparison_data_percentage_scenario,
    models_comparison_learning_rate_scenario,
    models_comparison_prediction_error_scenario,
    train_loss_curve_AND_test_loss_curve_scenario,
)
from utils import z_score_scaler
import pandas as pd
import numpy as np

raw_data = pd.read_csv("LinearRegression/DataSets/Boston_Housing_Dataset.csv")
raw_data = raw_data.drop(columns=["Unnamed: 0"])
T = raw_data["medv"].values
X = raw_data.drop(columns=["medv"]).values

X_Scaled = z_score_scaler(X)
data_with_bias = np.hstack([np.ones((X.shape[0], 1)), X_Scaled])

effect_of_sample_size_scenario(data_with_bias=data_with_bias, label=T)

effect_of_learning_rate_scenario(data_with_bias=data_with_bias, label=T)

train_loss_curve_AND_test_loss_curve_scenario(data_with_bias=data_with_bias, label=T)

models_comparison_learning_rate_scenario(data_with_bias=data_with_bias, label=T)

models_comparison_data_percentage_scenario(data_with_bias=data_with_bias, label=T)
model_comparison_prediction_scenario(data_with_bias=data_with_bias, label=T)
models_comparison_prediction_error_scenario(data_with_bias=data_with_bias, label=T)
