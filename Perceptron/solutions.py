from matplotlib import pyplot as plt
import numpy as np
from models import perceptron_from_sklearn, perceptron_my_model
from modelRunners import (
    model_runner_for_each_data_percentage,
    model_runner_for_each_learning_rate,
)
from utils import plotter


EPOCHS = 100
LR = 0.001


def effect_of_sample_size_scenario(weightedInput, label):
    (
        model_train_accuracy,
        model_train_precision,
        model_train_recall,
        model_train_f1,
        model_test_accuracy,
        model_test_precision,
        model_test_recall,
        model_test_f1,
        x_values,
    ) = model_runner_for_each_data_percentage(
        data=weightedInput,
        label=label,
        lr=LR,
        epoch=EPOCHS,
        model_function=perceptron_my_model,
    )
    plotter(
        values_arr=[model_train_accuracy, model_test_accuracy],
        line_label_arr=["Train accuracy", "Test accuracy"],
        x_values=x_values,
        x_label="Data%",
        y_label="Loss value",
        title="Sample Size Effect (My Perceptron)",
        show_percentage_for_x=True,
        show_values_for_each=True,
        file_name="Sample Size Effect",
    )


def effect_of_learning_rate_scenario(weightedInput, label):
    (
        model_train_accuracy,
        model_train_precision,
        model_train_recall,
        model_train_f1,
        model_test_accuracy,
        model_test_precision,
        model_test_recall,
        model_test_f1,
        x_values,
    ) = model_runner_for_each_learning_rate(
        data=weightedInput,
        label=label,
        epoch=EPOCHS,
        model_function=perceptron_my_model,
    )
    plotter(
        x_values=x_values,
        values_arr=[
            model_train_accuracy,
            model_test_accuracy,
            model_train_recall,
            model_test_recall,
        ],
        line_label_arr=[
            "Train accuracy ",
            "Test accuracy ",
            "train recall",
            "test recall",
        ],
        x_label="Learning Rate",
        y_label="Train Loss",
        title="Learning Rate Effect On Losses (My Perceptron)",
        show_percentage_for_x=False,
        show_values_for_each=True,
        file_name="Learning Rate Effect",
    )


def models_comparison_learning_rate_scenario(weightedInput, label):

    (
        model_train_accuracy,
        model_train_precision,
        model_train_recall,
        model_train_f1,
        model_test_accuracy,
        model_test_precision,
        model_test_recall,
        model_test_f1,
        x_values,
    ) = model_runner_for_each_learning_rate(
        data=weightedInput,
        label=label,
        epoch=EPOCHS,
        model_function=perceptron_my_model,
    )
    (
        train_accuracy,
        train_precision,
        train_recall,
        train_f1,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        x_values,
    ) = model_runner_for_each_learning_rate(
        data=weightedInput,
        label=label,
        epoch=EPOCHS,
        model_function=perceptron_from_sklearn,
    )
    plotter(
        x_values=x_values,
        values_arr=[
            model_train_accuracy,
            model_test_accuracy,
            train_accuracy,
            test_accuracy,
            model_train_precision,
            model_test_precision,
            train_precision,
            test_precision,
        ],
        line_label_arr=[
            "My Perceptron Train acc",
            "My Perceptron Test acc",
            "Sk Perceptron Test acc",
            "Sk Perceptron Train acc",
            "My Perceptron Train perc",
            "My Perceptron Test perc",
            "Sk Perceptron Train prec",
            "Sk Perceptron Test perc",
        ],
        x_label="Learning Rate",
        y_label="Loss",
        show_values_for_each=True,
        title="Learning Rate Model Comparison",
        show_percentage_for_x=False,
        file_name="Learning Rate Model Comparison",
    )


def models_comparison_data_percentage_scenario(weightedInput, label):
    (
        model_train_accuracy,
        model_train_precision,
        model_train_recall,
        model_train_f1,
        model_test_accuracy,
        model_test_precision,
        model_test_recall,
        model_test_f1,
        x_values,
    ) = model_runner_for_each_data_percentage(
        data=weightedInput,
        label=label,
        epoch=EPOCHS,
        lr=LR,
        model_function=perceptron_my_model,
    )
    (
        train_accuracy,
        train_precision,
        train_recall,
        train_f1,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        x_values,
    ) = model_runner_for_each_data_percentage(
        data=weightedInput,
        label=label,
        epoch=EPOCHS,
        lr=LR,
        model_function=perceptron_from_sklearn,
    )
    plotter(
        x_values=x_values,
        values_arr=[
            model_train_accuracy,
            model_test_accuracy,
            train_accuracy,
            test_accuracy,
        ],
        line_label_arr=[
            "My Perceptron Train acc",
            "My Perceptron Test acc",
            "Sk Regressor Train acc",
            "Sk Regressor Test acc",
        ],
        x_label="Learning Rate",
        y_label="Loss",
        show_values_for_each=True,
        title="Data Percentage Model Comparison",
        show_percentage_for_x=True,
        file_name="Data Percentage Model Comparison",
    )


# plotter(
#     x_values=np.arange(0, 20, 1),
#     values_arr=[
#         model.train_FP,
#         model.train_FN,
#         model.train_TP,
#         model.train_TN,
#         model.test_FP,
#         model.test_FN,
#         model.test_TP,
#         model.test_TN,
#     ],
#     line_label_arr=[
#         "Train FP",
#         "Train FN",
#         "Train TP",
#         "Train TN",
#         "Test FP",
#         "Test FN",
#         "Test TP",
#         "Test TN",
#     ],
#     x_label="Epochs",
#     y_label="Value",
#     file_name="Test",
#     title="Test",
#     show_values_for_each=True,
#     show_percentage_for_x=False,
# )
