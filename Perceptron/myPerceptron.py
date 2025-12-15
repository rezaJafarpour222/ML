import numpy as np


class Perceptron:

    def __init__(self, features):
        self.W = np.zeros(features)
        self.total_train_loss = []
        self.total_test_loss = []

    def predict(self, x):
        if (np.dot(self.W, x)) > 0:
            return 1
        else:
            return -1

    def loss_function(self, X, Y):
        prediction = X.dot(self.W)
        misclassified = (Y * prediction) < 0
        return np.sum(misclassified)

    def SGD(self, X_train, Y_train, X_test, Y_test, lr, epochs):
        indices = np.arange(X_train.shape[0])
        for _ in range(epochs):

            rand_idx = np.random.permutation(indices)
            X_train_shuffled = X_train[rand_idx]
            Y_train_shuffled = Y_train[rand_idx]

            for index in range(len(indices)):
                x_i = X_train_shuffled[index]
                t_i = Y_train_shuffled[index]
                if t_i * self.predict(x_i) <= 0:
                    gradient = -(x_i * t_i)
                    self.W -= lr * gradient

            train_loss = self.loss_function(X_train, Y_train)
            test_loss = self.loss_function(X_test, Y_test)
            self.total_test_loss.append(test_loss)
            self.total_train_loss.append(train_loss)
