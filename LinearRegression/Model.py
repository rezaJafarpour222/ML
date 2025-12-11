import numpy as np
from sklearn.metrics import mean_squared_error


class LinearRegressor:
    def __init__(self, numberofFeatures):
        self.W = np.zeros(numberofFeatures)
        self.total_train_lost = []
        self.total_test_lost = []

    def predict(self, X):
        return X.dot(self.W)

    def MSE(self, t, y_prediction):
        return np.mean((t - y_prediction) ** 2)
        # return mean_squared_error(t, y_prediction)

    def SGD(self, X_train_shuffled, Y_train_shuffled, X_test, Y_test, lr, epochs):
        indices = np.arange(X_train_shuffled.shape[0])

        for _ in range(epochs):
            rand_idx = np.random.permutation(indices)
            X_train_shuffled = X_train_shuffled[rand_idx]
            Y_train_shuffled = Y_train_shuffled[rand_idx]
            for index in range(len(indices)):
                x_i = X_train_shuffled[index].reshape(1, -1)
                t_i = Y_train_shuffled[index]

                y_pred_i = self.predict(x_i)
                error_i = t_i - y_pred_i

                gradient = -x_i.T.dot(error_i)
                self.W -= lr * gradient

            y_train_pred = self.predict(X_train_shuffled)
            y_test_pred = self.predict(X_test)
            train_loss = self.MSE(Y_train_shuffled, y_train_pred)
            test_loss = self.MSE(Y_test, y_test_pred)
            self.total_train_lost.append(train_loss)
            self.total_test_lost.append(test_loss)

        return (self.total_train_lost, self.total_test_lost)
