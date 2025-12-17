from myPerceptron import Perceptron
from sklearn.linear_model import Perceptron as skPerceptron
from utils import splitter
from evaluation import f1_measure, accuracy, loss_calculator, precision, recall


def perceptron_from_sklearn(data, label, lr, epochs, splitPercent):

    (X_train, Y_train, X_test, Y_test) = splitter(
        data=data, label=label, splitPrecent=splitPercent
    )
    model = skPerceptron(
        max_iter=epochs, eta0=lr, learning_rate="constant", warm_start=True
    )
    train_TPs, train_TNs, train_FPs, train_FNs = []
    test_TPs, test_TNs, test_FPs, test_FNs = []
    for _ in range(epochs):
        model.fit(X_train, Y_train)
        train_pred = model.fit(X_train)
        test_pred = model.fit(X_test)
        (train_TP, train_TN, train_FP, train_FN) = loss_calculator(train_pred, Y_train)
        (test_TP, test_TN, test_FP, test_FN) = loss_calculator(test_pred, Y_test)

    train_acc = accuracy(
        TP=model.train_TP, TN=model.train_TN, FP=model.train_FP, FN=model.train_FP
    )
    train_recall = recall(TP=model.train_TP, FN=model.train_FN)
    train_precision = precision(TP=model.train_TP, FN=model.train_FN)
    train_f1_measure = f1_measure(precision=train_precision, recall=train_recall)

    test_acc = accuracy(
        TP=model.test_TP, TN=model.test_TN, FP=model.test_FP, FN=model.test_FP
    )
    test_recall = recall(TP=model.test_TP, FN=model.test_FN)
    test_precision = precision(TP=model.test_TP, FN=model.test_FN)
    test_f1_measure = f1_measure(precision=test_precision, recall=test_recall)

    return (
        train_acc,
        test_acc,
        train_recall,
        test_recall,
        train_precision,
        test_precision,
        train_f1_measure,
        test_f1_measure,
    )


def perceptron_my_model(data, label, lr, epochs, splitPercent):

    (X_train, Y_train, X_test, Y_test) = splitter(
        data=data, label=label, splitPrecent=splitPercent
    )
    model = Perceptron(data.shape[1])
    model.SGD(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        lr=lr,
        epochs=epochs,
    )
    train_acc = accuracy(
        TP=model.train_TP, TN=model.train_TN, FP=model.train_FP, FN=model.train_FP
    )
    train_recall = recall(TP=model.train_TP, FN=model.train_FN)
    train_precision = precision(TP=model.train_TP, FN=model.train_FN)
    train_f1_measure = f1_measure(precision=train_precision, recall=train_recall)

    test_acc = accuracy(
        TP=model.test_TP, TN=model.test_TN, FP=model.test_FP, FN=model.test_FP
    )
    test_recall = recall(TP=model.test_TP, FN=model.test_FN)
    test_precision = precision(TP=model.test_TP, FN=model.test_FN)
    test_f1_measure = f1_measure(precision=test_precision, recall=test_recall)

    return (
        train_acc,
        test_acc,
        train_recall,
        test_recall,
        train_precision,
        test_precision,
        train_f1_measure,
        test_f1_measure,
    )
