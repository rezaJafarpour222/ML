from typing import Literal
from unittest import TestCase
import algorithms
from util import plotter


def SVM_metrics(weightedInput, label, kernel, fileName, C=1):

    (
        train_acc,
        test_acc,
        train_recall,
        test_recall,
        train_precision,
        test_precision,
        train_f1_measure,
        test_f1_measure,
    ) = algorithms.SVM_model(data=weightedInput, label=label, kernel=kernel, C=C)
    plotter(
        values_arr=[
            train_acc,
            train_recall,
            train_precision,
            test_acc,
            test_recall,
            test_precision,
        ],
        label_arr=[
            "Train Accuracy",
            "Train Recall",
            "Train Precision",
            "Test Accuracy",
            "Test Recall",
            "Test Precision",
        ],
        file_name=fileName,
        y_label="Score",
        width=0.2,
    )
    return (test_acc, test_precision, test_recall)


def LDA_metrics(
    weightedInput, label, fileName, solver: Literal["svd", "isqr", "eigen"]
):
    (
        train_acc,
        test_acc,
        train_recall,
        test_recall,
        train_precision,
        test_precision,
        train_f1_measure,
        test_f1_measure,
    ) = algorithms.LDA_model(data=weightedInput, label=label, solver=solver)
    plotter(
        values_arr=[
            train_acc,
            train_recall,
            train_precision,
            test_acc,
            test_recall,
            test_precision,
        ],
        label_arr=[
            "Train Accuracy",
            "Train Recall",
            "Train Precision",
            "Test Accuracy",
            "Test Recall",
            "Test Precision",
        ],
        file_name=fileName,
        y_label="Score",
        width=0.2,
    )

    return (test_acc, test_precision, test_recall)


def DecisionTree_metrics(
    weightedInput,
    label,
    max_depth,
    fileName,
    sample_leaf=5,
    sample_split=5,
    criterion: Literal["gini", "entropy", "log_loss"] = "entropy",
):
    (
        train_acc,
        test_acc,
        train_recall,
        test_recall,
        train_precision,
        test_precision,
        train_f1_measure,
        test_f1_measure,
    ) = algorithms.DecisionTree_model(
        data=weightedInput,
        label=label,
        max_depth=max_depth,
        samples_leaf=sample_leaf,
        samples_split=sample_split,
        criterion=criterion,
    )
    plotter(
        values_arr=[
            train_acc,
            train_recall,
            train_precision,
            test_acc,
            test_recall,
            test_precision,
        ],
        label_arr=[
            "Train Accuracy",
            "Train Recall",
            "Train Precision",
            "Test Accuracy",
            "Test Recall",
            "Test Precision",
        ],
        file_name=fileName,
        y_label="Score",
        width=0.2,
    )

    return (test_acc, test_precision, test_recall)


