import numpy as np
from myPerceptron import Perceptron
from models import perceptron_from_sklearn, perceptron_my_model
from modelRunners import (
    model_runner_for_each_data_percentage,
    model_runner_for_each_learning_rate,
)
from utils import plotter, splitter


EPOCHS = 50
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
    # step = int(EPOCHS / EPOCHS)
    plotter(
        values_arr=[
            model_train_accuracy,
            model_test_accuracy,
            model_train_recall,
            model_test_recall,
            model_train_precision,
            model_test_precision,
            model_train_f1,
            model_test_f1,
        ],
        line_label_arr=[
            "Train accuracy ",
            "Test accuracy ",
            "Train recall",
            "Test recall",
            "Train precision",
            "Test precision",
            "Train f1",
            "Test f1",
        ],
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
            model_train_precision,
            model_test_precision,
            model_train_f1,
            model_test_f1,
        ],
        line_label_arr=[
            "Train accuracy ",
            "Test accuracy ",
            "Train recall",
            "Test recall",
            "Train precision",
            "Test precision",
            "Train f1",
            "Test f1",
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
        _,
        _,
        _,
        _,
        model_test_accuracy,
        model_test_precision,
        model_test_recall,
        _,
        x_values,
    ) = model_runner_for_each_learning_rate(
        data=weightedInput,
        label=label,
        epoch=EPOCHS,
        model_function=perceptron_my_model,
    )
    (
        _,
        _,
        _,
        _,
        test_accuracy,
        test_precision,
        test_recall,
        _,
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
            model_test_precision,
            model_test_accuracy,
            model_test_recall,
            test_precision,
            test_accuracy,
            test_recall,
        ],
        line_label_arr=[
            "My  precision",
            "My  accurracy",
            "My  recall",
            "Sk  precision",
            "Sk  accuracy",
            "Sk  recall",
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
        _,
        _,
        _,
        _,
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
        _,
        _,
        _,
        _,
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
            model_test_recall,
            model_test_f1,
            test_accuracy,
            test_precision,
            test_recall,
            test_f1,
        ],
        line_label_arr=[
            "my accuracy",
            "my precision",
            "my recall",
            "my f1",
            "sk accuracy",
            "sk precision",
            "sk recall",
            "sk f1",
        ],
        x_label="Data percentage",
        y_label="Loss",
        show_values_for_each=True,
        title="Data Percentage Model Comparison",
        show_percentage_for_x=True,
        file_name="Data Percentage Model Comparison",
    )


def metrics_scenario(weightedInput, label):
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
    step = int(EPOCHS / 10)

    plotter(
        values_arr=[
            model.train_accuracies[::step],
            model.test_accuracies[::step],
            model.train_precisions[::step],
            model.test_precisions[::step],
            model.train_recalls[::step],
            model.test_recalls[::step],
            model.train_f1_measures[::step],
            model.test_f1_measures[::step],
        ],
        line_label_arr=[
            "Train accuracy ",
            "Test accuracy ",
            "Train precision",
            "Test precision",
            "Train recall",
            "Test recall",
            "Train f1",
            "Test f1",
        ],
        x_values=np.arange(0, EPOCHS, step),
        x_label="Epoch",
        y_label="Loss value",
        title=" Metrics(My Perceptron)",
        show_percentage_for_x=False,
        show_values_for_each=True,
        file_name=" Metrics",
    )
