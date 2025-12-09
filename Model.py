import numpy as np


class LinearRegressor:
    def __init__(self, numberofFeatures):
        self.W = np.zeros(numberofFeatures)
        self.total_train_lost = []
        self.total_test_lost = []

    def predict(self, X):
        return X.dot(self.W)

    def SSE(self, t, y_prediction):
        return np.mean((t - y_prediction) ** 2)

    def GD(self, X_train, Y_train, X_test, Y_test, lr, epochs):

        for i in range(epochs):
            y_pred = self.predict(X_train)
            error = Y_train - y_pred
            gradient = -2 * X_train.T.dot(error)
            # gradient = -2 / n_samples * X_train.T.dot(error)
            self.W -= lr * gradient
            trainSSE = self.SSE(Y_train, self.predict(X_train))
            self.total_train_lost.append(trainSSE)
            testSSE = self.SSE(Y_test, self.predict(X_test))
            self.total_test_lost.append(testSSE)

    def SGD(self, X_train_shuffled, Y_train_shuffled, X_test, Y_test, lr, epochs):
        n_samples = X_train_shuffled.shape[0]
        indices = np.arange(n_samples)

        for _ in range(epochs):
            np.random.shuffle(indices)
            X_train_shuffled = X_train_shuffled[indices]
            Y_train_shuffled = Y_train_shuffled[indices]
            for index in range(n_samples):
                x_i = X_train_shuffled[index].reshape(1, -1)
                t_i = Y_train_shuffled[index]

                y_pred_i = self.predict(x_i)
                error_i = t_i - y_pred_i

                gradient = -2 * x_i.T.dot(error_i)
                self.W -= lr * gradient

            y_train_pred = self.predict(X_train_shuffled)
            y_test_pred = self.predict(X_test)
            train_loss = self.SSE(Y_train_shuffled, y_train_pred)
            test_loss = self.SSE(Y_test, y_test_pred)
            self.total_train_lost.append(train_loss)
            self.total_test_lost.append(test_loss)

        return (self.total_train_lost, self.total_test_lost)
