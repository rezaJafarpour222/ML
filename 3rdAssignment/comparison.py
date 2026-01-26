import algorithms
from util import plotter


def SVM_metrics(weightedInput, label):

    (
        train_acc,
        test_acc,
        train_recall,
        test_recall,
        train_precision,
        test_precision,
        train_f1_measure,
        test_f1_measure,
    ) = algorithms.SVM_model(data=weightedInput, label=label, splitPercentage=1.0)
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
        file_name="SVM",
        y_label="Score",
        title="SVM Metrics",
        width=0.2,
    )


def LDA_metrics(weightedInput, label):
    (
        train_acc,
        test_acc,
        train_recall,
        test_recall,
        train_precision,
        test_precision,
        train_f1_measure,
        test_f1_measure,
    ) = algorithms.LDA_model(data=weightedInput, label=label, splitPercentage=1.0)
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
        file_name="LDA",
        y_label="Score",
        title="LDA Metrics",
        width=0.2,
    )


def DecisionTree_metrics(weightedInput, label):
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
        data=weightedInput, label=label, splitPercentage=1.0
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
        file_name="Decision Tree",
        y_label="Score",
        title="Decision Tree Metrics",
        width=0.2,
    )
