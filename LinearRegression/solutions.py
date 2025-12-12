from matplotlib import pyplot as plt
import numpy as np

from modelRunners import (
    model_runner_for_each_data_percentage,
    model_runner_for_each_learning_rate,
)
from models import linearRegressor_from_sklearn, linearRegressor_my_model
import myRegressor
from utils import plotter, splitter

EPOCHS = 20
LR = 0.001


def train_loss_curve_AND_test_loss_curve_scenario(data_with_bias, label):
    (X_train, Y_train, X_test, Y_test) = splitter(
        data=data_with_bias, label=label, splitPrecent=1.0
    )

    model = myRegressor.LinearRegressor(data_with_bias.shape[1])
    (_, _) = model.SGD(
        X_train_shuffled=X_train,
        Y_train_shuffled=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        lr=LR,
        epochs=EPOCHS,
    )
    train_min = np.min(model.total_train_lost)
    test_min = np.min(model.total_test_lost)
    arg_train_min = np.argmin(model.total_train_lost)
    arg_test_min = np.argmin(model.total_test_lost)
    print(arg_train_min, " ", train_min)
    print(arg_test_min, " ", test_min)
    step = int(EPOCHS / 10)
    plotter(
        x_values=np.arange(0, EPOCHS, step),
        first=model.total_train_lost[::step],
        second=model.total_test_lost[::step],
        first_line_title="Train Loss",
        second_line_title="Test Loss",
        x_label="Epochs",
        y_label="Loss",
        title="Train and Test Loss Curve (My Regressor)",
        show_percentage_for_x=False,
        show_values_for_each=True,
        file_name="Losses_curve",
    )


def effect_of_sample_size_scenario(data_with_bias, label):
    (train_loss, test_loss, x_values) = model_runner_for_each_data_percentage(
        data=data_with_bias,
        label=label,
        lr=LR,
        epoch=EPOCHS,
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


def effect_of_learning_rate_scenario(data_with_bias, label):
    (train_loss, test_loss, x_values) = model_runner_for_each_learning_rate(
        data=data_with_bias,
        label=label,
        lr=LR,
        epoch=EPOCHS,
        model_function=linearRegressor_my_model,
    )
    plotter(
        x_values=x_values,
        first=train_loss,
        second=test_loss,
        first_line_title="Train Loss",
        second_line_title="Test Loss",
        x_label="Learning Rate",
        y_label="Train Loss",
        title="Learning Rate Effect On Losses (My Regressor)",
        show_percentage_for_x=False,
        show_values_for_each=True,
        file_name="Learning Rate Effect",
    )


def models_comparison_learning_rate_scenario(data_with_bias, label):
    (my_train_loss, my_test_loss, x_values) = model_runner_for_each_learning_rate(
        data=data_with_bias,
        label=label,
        lr=LR,
        epoch=EPOCHS,
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


def models_comparison_data_percentage_scenario(data_with_bias, label):
    (my_train_loss, my_test_loss, x_values) = model_runner_for_each_data_percentage(
        data=data_with_bias,
        label=label,
        lr=LR,
        epoch=EPOCHS,
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


def model_comparison_prediction_scenario(data_with_bias, label):
    (_, _, sk_pred, _) = linearRegressor_from_sklearn(
        data=data_with_bias, label=label, lr=LR, epochs=EPOCHS, splitPercent=1.0
    )
    (_, _, pred, y) = linearRegressor_my_model(
        data=data_with_bias, label=label, lr=LR, epochs=EPOCHS, splitPercent=1.0
    )
    plt.figure(figsize=(16, 9), dpi=500)
    plt.scatter(range(len(y)), y, color="blue", marker="o", label="True values")

    plt.scatter(
        range(len(sk_pred)),
        sk_pred,
        color="red",
        marker="x",
        alpha=1.0,
        label="Sk Regressor",
    )
    plt.scatter(
        range(len(pred)),
        pred,
        color="green",
        marker="+",
        alpha=1.0,
        label="My Regressor",
    )
    plt.legend()
    plt.title("True values vs predicted values")
    plt.xlabel("Data Point")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"LinearRegression/plots/Prediction", dpi=500)
    plt.close()


def models_comparison_prediction_error_scenario(data_with_bias, label):
    (_, _, sk_pred, _) = linearRegressor_from_sklearn(
        data=data_with_bias, label=label, lr=LR, epochs=EPOCHS, splitPercent=1.0
    )
    (_, _, pred, y) = linearRegressor_my_model(
        data=data_with_bias, label=label, lr=0.001, epochs=500, splitPercent=1.0
    )
    sk_pred_error = y - sk_pred
    pred_error = y - pred
    plt.figure(figsize=(16, 9), dpi=500)
    plt.scatter(
        range(len(sk_pred_error)),
        sk_pred_error,
        color="red",
        marker="x",
        label="Sk Regressor",
        alpha=0.8,
    )
    plt.scatter(
        range(len(pred_error)),
        pred_error,
        color="green",
        marker="o",
        label="My Regressor",
        alpha=0.8,
    )
    plt.axhline(0, color="blue", linestyle="-", label="Perfect Prediction Line")
    plt.xlabel("Data Point")
    plt.ylabel("Error (True - Prediction)")
    plt.legend()
    plt.title("True values vs predicted values")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"LinearRegression/plots/Prediction Error", dpi=500)
    plt.close()
