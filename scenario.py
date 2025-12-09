import numpy as np
import Model
from training import (
    linearRegressor_from_sklearn,
    linearRegressor_my_model,
    model_runner_for_each_data_percentage,
    model_runner_for_each_learning_rate,
)
from utils import plotter, splitter


def train_loss_curve_AND_test_loss_curve_scenario(data_with_bias, label):
    (X_train, Y_train, X_test, Y_test) = splitter(
        data=data_with_bias, label=label, splitPrecent=1.0
    )

    model = Model.LinearRegressor(data_with_bias.shape[1])
    (_, _) = model.SGD(
        X_train_shuffled=X_train,
        Y_train_shuffled=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        lr=0.001,
        epochs=500,
    )
    train_min = np.min(model.total_train_lost)
    test_min = np.min(model.total_test_lost)
    arg_train_min = np.argmin(model.total_train_lost)
    arg_test_min = np.argmin(model.total_test_lost)
    print(arg_train_min, " ", train_min)
    print(arg_test_min, " ", test_min)

    plotter(
        x_values=np.arange(0, 500, 49),
        first=model.total_train_lost[::49],
        second=model.total_test_lost[::49],
        first_line_title="Train Loss",
        second_line_title="Second Loss",
        x_label="Epochs",
        y_label="Loss",
        title="Train and Test Loss Curve (My Regressor)",
        show_percentage_for_x=False,
        show_values_for_each=True,
        file_name="Losses_curve",
    )


def sample_size_effect_scenario(data_with_bias, label):
    (train_loss, test_loss, x_values) = model_runner_for_each_data_percentage(
        data=data_with_bias,
        label=label,
        lr=0.001,
        epoch=500,
        model_function=linearRegressor_my_model,
    )
    plotter(
        x_values=x_values,
        first=train_loss,
        second=test_loss,
        first_line_title="Train Loss",
        second_line_title="Test Loss",
        x_label="Data%",
        y_label="Train Loss",
        title="Sample Size Effect (My Regressor)",
        show_percentage_for_x=True,
        show_values_for_each=True,
        file_name="Sample Size Effect",
    )


def learning_rate_effect_scenario(data_with_bias, label):
    (train_loss, test_loss, x_values) = model_runner_for_each_learning_rate(
        data=data_with_bias,
        label=label,
        lr=0.001,
        epoch=500,
        model_function=linearRegressor_my_model,
    )
    plotter(
        x_values=x_values,
        first=train_loss,
        second=test_loss,
        first_line_title="Train Loss",
        second_line_title="Test Loss",
        x_label="Learning Rate",
        y_label="train Loss",
        title="Learning Rate Effect On Losses (My Regressor)",
        show_percentage_for_x=False,
        show_values_for_each=True,
        file_name="Learning Rate Effect",
    )


def comparison_learning_rate_scenario(data_with_bias, label):
    (my_train_loss, my_test_loss, x_values) = model_runner_for_each_learning_rate(
        data=data_with_bias,
        label=label,
        lr=0.001,
        epoch=500,
        model_function=linearRegressor_my_model,
    )

    (sk_train_loss, sk_test_loss, _) = model_runner_for_each_learning_rate(
        data=data_with_bias,
        label=label,
        lr=0.001,
        epoch=500,
        model_function=linearRegressor_from_sklearn,
    )
    plotter(
        x_values=x_values,
        first=my_train_loss,
        second=sk_train_loss,
        first_line_title="My Regressor Train Loss",
        second_line_title="Sk Regressor Train Loss",
        x_label="Learning Rate",
        y_label="Train Loss",
        title="Learning Rate Effect On Losses Comparison (Train Loss)",
        show_percentage_for_x=False,
        show_values_for_each=True,
        file_name="Learning Rate Model Comparison (Train Loss)",
    )
    plotter(
        x_values=x_values,
        first=my_test_loss,
        second=sk_test_loss,
        first_line_title="My Regressor Test Loss",
        second_line_title="Sk Regressor Test Loss",
        x_label="Learning Rate",
        y_label="Test Loss",
        title="Learning Rate Effect On Losses Comparison (Test Loss)",
        show_percentage_for_x=False,
        show_values_for_each=True,
        file_name="Learning Rate Model Comparison (Test Loss)",
    )


def comparison_data_percentage_scenario(data_with_bias, label):
    (my_train_loss, my_test_loss, x_values) = model_runner_for_each_data_percentage(
        data=data_with_bias,
        label=label,
        lr=0.001,
        epoch=500,
        model_function=linearRegressor_my_model,
    )

    (sk_train_loss, sk_test_loss, _) = model_runner_for_each_data_percentage(
        data=data_with_bias,
        label=label,
        lr=0.001,
        epoch=500,
        model_function=linearRegressor_from_sklearn,
    )
    plotter(
        x_values=x_values,
        first=my_train_loss,
        second=sk_train_loss,
        first_line_title="My Regressor Train Loss",
        second_line_title="Sk Regressor Train Loss",
        x_label="Data Percentage",
        y_label="Train Loss",
        title="Data Percentage Effect On Losses Comparison (Train Loss)",
        show_percentage_for_x=True,
        show_values_for_each=True,
        file_name="Data Percentage Model Comparison Train Loss",
    )
    plotter(
        x_values=x_values,
        first=my_test_loss,
        second=sk_test_loss,
        first_line_title="My Regressor Test Loss",
        second_line_title="Sk Regressor Test Loss",
        x_label="Data percentage",
        y_label="Test Loss",
        title="Data Percentage Effect On Losses Comparison (Test Loss)",
        show_percentage_for_x=True,
        show_values_for_each=True,
        file_name="Data Percentage Model Comparison Test Loss",
    )
