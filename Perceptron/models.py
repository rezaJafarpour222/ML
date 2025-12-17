from myPerceptron import Perceptron
from sklearn.linear_model import Perceptron as skPerceptron
from utils import splitter
from evaluation import f1_measure, accuracy, loss_calculator, precision, recall


def perceptron_from_sklearn(data, label, lr, epochs, splitPercent):

    (X_train, Y_train, X_test, Y_test) = splitter(
        data=data, label=label, splitPrecent=splitPercent
    )
    model = skPerceptron(max_iter=epochs, eta0=lr, tol=None)
    model.fit(X_train, Y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    (train_TP, train_TN, train_FP, train_FN) = loss_calculator(train_pred, Y_train)
    (test_TP, test_TN, test_FP, test_FN) = loss_calculator(test_pred, Y_test)

    train_acc = accuracy(TP=train_TP, TN=train_TN, FP=train_FP, FN=train_FN)
    train_recall = recall(TP=train_TP, FN=train_FN)
    train_precision = precision(TP=train_TP, FP=train_FP)
    train_f1_measure = f1_measure(precision=train_precision, recall=train_recall)

    test_acc = accuracy(TP=test_TP, TN=test_TN, FP=test_FP, FN=test_FN)

    test_recall = recall(TP=test_TP, FN=test_FN)
    test_precision = precision(TP=test_TP, FP=test_FP)
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

    return (
        model.train_accuracies[epochs - 1],
        model.test_accuracies[epochs - 1],
        model.train_recalls[epochs - 1],
        model.test_recalls[epochs - 1],
        model.train_precisions[epochs - 1],
        model.test_precisions[epochs - 1],
        model.train_f1_measures[epochs - 1],
        model.test_f1_measures[epochs - 1],
    )
