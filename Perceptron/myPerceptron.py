import numpy as np

from evaluation import accuracy, f1_measure, precision, recall


class Perceptron:

    def __init__(self, features):
        self.W = np.zeros(features)
        self.train_accuracies = []
        self.train_recalls = []
        self.train_precisions = []
        self.train_f1_measures = []
        self.test_accuracies = []
        self.test_recalls = []
        self.test_precisions = []
        self.test_f1_measures = []

    def predict_for_one(self, x):
        if (np.dot(self.W, x)) > 0:
            return 1
        else:
            return -1

    def loss_calculator(self, X, Y):
        prediction = X.dot(self.W)
        TP = np.sum((prediction > 0) & (Y > 0))
        TN = np.sum((prediction < 0) & (Y < 0))
        FP = np.sum((prediction > 0) & (Y < 0))
        FN = np.sum((prediction < 0) & (Y > 0))
        return (TP, TN, FP, FN)

    def eval_calculator(
        self, test_TP, test_TN, test_FP, test_FN, train_TP, train_TN, train_FP, train_FN
    ):
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
            train_recall,
            train_precision,
            train_f1_measure,
            test_acc,
            test_recall,
            test_precision,
            test_f1_measure,
        )

    def SGD(self, X_train, Y_train, X_test, Y_test, lr, epochs):
        indices = np.arange(X_train.shape[0])
        for _ in range(epochs):

            rand_idx = np.random.permutation(indices)
            X_train_shuffled = X_train[rand_idx]
            Y_train_shuffled = Y_train[rand_idx]

            for index in range(len(indices)):
                x_i = X_train_shuffled[index]
                t_i = Y_train_shuffled[index]
                is_update_needed = t_i * self.predict_for_one(x_i)
                if is_update_needed <= 0:
                    gradient = -(x_i * t_i)
                    self.W -= lr * gradient

            (train_TP, train_TN, train_FP, train_FN) = self.loss_calculator(
                X_train, Y_train
            )
            (test_TP, test_TN, test_FP, test_FN) = self.loss_calculator(X_test, Y_test)
            (
                train_acc,
                train_recall,
                train_precision,
                train_f1_measure,
                test_acc,
                test_recall,
                test_precision,
                test_f1_measure,
            ) = self.eval_calculator(
                test_TP,
                test_TN,
                test_FP,
                test_FN,
                train_TP,
                train_TN,
                train_FP,
                train_FN,
            )

            self.train_accuracies.append(train_acc)
            self.train_precisions.append(train_precision)
            self.train_recalls.append(train_recall)
            self.train_f1_measures.append(train_f1_measure)

            self.test_accuracies.append(test_acc)
            self.test_precisions.append(test_precision)
            self.test_recalls.append(test_recall)
            self.test_f1_measures.append(test_f1_measure)
