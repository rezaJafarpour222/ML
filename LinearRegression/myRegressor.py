import numpy as np


class LinearRegressor:
    def __init__(self, features):
        self.W = np.zeros(features)
        self.total_train_lost = []
        self.total_test_lost = []

    def predict(self, X):
        return X.dot(self.W)

    def MSE(self, t, y_prediction):
        return np.mean((t - y_prediction) ** 2)

    def SGD(self, X_train, Y_train, X_test, Y_test, lr, epochs):
        indices = np.arange(X_train.shape[0])

        for _ in range(epochs):
            rand_idx = np.random.permutation(indices)
            X_train = X_train[rand_idx]
            Y_train = Y_train[rand_idx]

            for index in range(len(indices)):
                x_i = X_train[index].reshape(1, -1)
                t_i = Y_train[index]

                y_pred_i = self.predict(x_i)

                error_i = t_i - y_pred_i
                gradient = -x_i.T.dot(error_i)
                self.W -= lr * gradient

            y_train_pred = self.predict(X_train)
            y_test_pred = self.predict(X_test)

            train_loss = self.MSE(Y_train, y_train_pred)
            test_loss = self.MSE(Y_test, y_test_pred)

            self.total_train_lost.append(train_loss)
            self.total_test_lost.append(test_loss)

        return (self.total_train_lost, self.total_test_lost)
