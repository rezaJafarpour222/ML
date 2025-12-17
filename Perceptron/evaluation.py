import numpy as np


def loss_calculator(prediction, label):
    TP = np.sum((prediction > 0) & (label > 0))
    TN = np.sum((prediction < 0) & (label < 0))
    FP = np.sum((prediction > 0) & (label < 0))
    FN = np.sum((prediction < 0) & (label > 0))
    return (TP, TN, FP, FN)


def accuracy(TP, TN, FP, FN):
    trues = TP + TN
    all = TP + TN + FP + FN
    return trues / all


def recall(TP, FN):
    return TP / (TP + FN)


def precision(TP, TF):
    TP / (TF + TP)


def f1_measure(precision, recall):
    beta = 1
    numerator = ((beta**2) + 1) * precision * recall
    denomerator = ((beta**2) * precision) + recall
    return numerator / denomerator
