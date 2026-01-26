from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from metrics import accuracy, f1_measure, loss_calculator, precision, recall
from util import splitter


def SVM_model(data, label, splitPercentage):
    (X_train, Y_train, X_test, Y_test) = splitter(
        data=data, label=label, splitPrecent=splitPercentage
    )
    model = SVC(kernel="linear")
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


def LDA_model(data, label, splitPercentage):
    (X_train, Y_train, X_test, Y_test) = splitter(
        data=data, label=label, splitPrecent=splitPercentage
    )
    model = LinearDiscriminantAnalysis(solver="lsqr")
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


def DecisionTree_model(data, label, splitPercentage):
    (X_train, Y_train, X_test, Y_test) = splitter(
        data=data, label=label, splitPrecent=splitPercentage
    )
    model = DecisionTreeClassifier(
        # max_depth=5,
        # max_leaf_nodes=5,
        criterion="entropy",
        random_state=42,
    )
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
