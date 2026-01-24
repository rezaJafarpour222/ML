import numpy as np

from metrics import (
    accuracy,
    precision,
    loss_calculator,
    recall,
    f1_measure,
)


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
        return np.where(x.dot(self.W) > 0, 1, -1)

    def predict(self, X):
        return np.where(X.dot(self.W) > 0, 1, -1)

    def SGD(self, X_train, Y_train, X_test, Y_test, lr, epochs):
        indices = np.arange(X_train.shape[0])

        # pocket vars
        best_W = self.W.copy()
        best_recall = -1

        for _ in range(epochs):
            rand_idx = np.random.permutation(indices)
            X_train_shuffled = X_train[rand_idx]
            Y_train_shuffled = Y_train[rand_idx]

            for i in range(len(indices)):
                x_i = X_train_shuffled[i]
                y_i = Y_train_shuffled[i]

                if y_i * np.dot(self.W, x_i) <= 0:
                    self.W += lr * y_i * x_i

                    # pocketing part
                    train_pred = self.predict(X_train)
                    TP, TN, FP, FN = loss_calculator(train_pred, Y_train)
                    rec = recall(TP=TP, FN=FN)

                    if rec > best_recall:
                        best_recall = rec
                        best_W = self.W.copy()

            train_pred = self.predict(X_train)
            test_pred = self.predict(X_test)

            train_TP, train_TN, train_FP, train_FN = loss_calculator(
                train_pred, Y_train
            )
            test_TP, test_TN, test_FP, test_FN = loss_calculator(test_pred, Y_test)

            train_acc = accuracy(train_TP, train_TN, train_FP, train_FN)
            train_rec = recall(train_TP, train_FN)
            train_prec = precision(train_TP, train_FP)
            train_f1 = f1_measure(train_prec, train_rec)

            test_acc = accuracy(test_TP, test_TN, test_FP, test_FN)
            test_rec = recall(test_TP, test_FN)
            test_prec = precision(test_TP, test_FP)
            test_f1 = f1_measure(test_prec, test_rec)

            self.train_accuracies.append(train_acc)
            self.train_recalls.append(train_rec)
            self.train_precisions.append(train_prec)
            self.train_f1_measures.append(train_f1)

            self.test_accuracies.append(test_acc)
            self.test_recalls.append(test_rec)
            self.test_precisions.append(test_prec)
            self.test_f1_measures.append(test_f1)

        self.W = best_W
