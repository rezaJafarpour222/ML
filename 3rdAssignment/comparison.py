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


def models_comparison(weightedInput, label):
    (
        _,
        TREE_test_acc,
        _,
        TREE_test_recall,
        _,
        TREE_test_precision,
        _,
        _,
    ) = algorithms.DecisionTree_model(
        data=weightedInput, label=label, splitPercentage=1.0
    )
    (
        _,
        SVM_test_acc,
        _,
        SVM_test_recall,
        _,
        SVM_test_precision,
        _,
        _,
    ) = algorithms.SVM_model(data=weightedInput, label=label, splitPercentage=1.0)
    (
        _,
        LDA_test_acc,
        _,
        LDA_test_recall,
        _,
        LDA_test_precision,
        _,
        _,
    ) = algorithms.LDA_model(data=weightedInput, label=label, splitPercentage=1.0)
    plotter(
        values_arr=[
            TREE_test_acc,
            SVM_test_acc,
            LDA_test_acc,
            TREE_test_precision,
            SVM_test_precision,
            LDA_test_precision,
            TREE_test_recall,
            SVM_test_recall,
            LDA_test_recall,
        ],
        label_arr=[
            "Tree Accuracy",
            "SVM Accuracy",
            "LDA Accuracy",
            "Tree Recall",
            "SVM Recall",
            "LDA Recall",
            "Tree Precision",
            "SVM Precision",
            "LDA Precision",
        ],
        file_name="Models Comparison (Test)",
        y_label="Score",
        title="Models Comparison (Test)",
        width=0.2,
    )
