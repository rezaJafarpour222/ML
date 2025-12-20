from matplotlib import pyplot as plt
import numpy as np
from myPerceptron import Perceptron
from models import perceptron_from_sklearn, perceptron_my_model
from modelRunners import (
    model_runner_for_each_data_percentage,
    model_runner_for_each_learning_rate,
)
from utils import plotter, splitter


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
    step = EPOCHS / 100
    plotter(
        values_arr=[model_train_precision[::step], model_train_accuracy[::step]],
        line_label_arr=["precision", " accuracy"],
        x_values=x_values[::step],
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
            model_test_accuracy,
            model_test_precision,
            test_accuracy,
            train_precision,
        ],
        line_label_arr=[
            "My Perceptron Test acc",
            "My Perceptron Test prec",
            "Sk Regressor Test acc",
            "Sk Regressor Test prec",
        ],
        x_label="Learning Rate",
        y_label="Loss",
        show_values_for_each=True,
        title="Data Percentage Model Comparison",
        show_percentage_for_x=True,
        file_name="Data Percentage Model Comparison",
    )


def precision_accuracy_curves_scenario(weightedInput, label):
    (X_train, Y_train, X_test, Y_test) = splitter(
        data=weightedInput, label=label, splitPrecent=1.0
    )
    model = Perceptron(weightedInput.shape[1])
    model.SGD(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        lr=LR,
        epochs=EPOCHS,
    )
    step = int(EPOCHS / 100)

    plotter(
        values_arr=[model.test_recalls[::step], model.test_precisions[::step]],
        line_label_arr=["recall", " precision"],
        x_values=np.arange(0, EPOCHS, step),
        x_label="Data%",
        y_label="Loss value",
        title=" Effect (My Perceptron)",
        show_percentage_for_x=False,
        show_values_for_each=True,
        file_name=" Effect",
    )
