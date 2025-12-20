import numpy as np


def loss_calculator(prediction, label):
    TP = np.sum((prediction > 0) & (label > 0))
    TN = np.sum((prediction < 0) & (label < 0))
    FP = np.sum((prediction > 0) & (label < 0))
    FN = np.sum((prediction < 0) & (label > 0))
    return (TP, TN, FP, FN)


def accuracy(TP, TN, FP, FN):
    numerator = TP + TN
    denominator = TP + TN + FP + FN
    if denominator != 0:
        return numerator / denominator
    return 0


def recall(TP, FN):
    denominator = TP + FN
    if denominator != 0:
        return TP / (TP + FN)
    return 0


def precision(TP, FP):
    denominator = TP + FP
    if denominator != 0:
        return TP / denominator
    return 0


def f1_measure(precision, recall):
    beta = 1
    numerator = ((beta**2) + 1) * precision * recall
    denominator = ((beta**2) * precision) + recall
    if denominator != 0:
        return numerator / denominator
    return 0
