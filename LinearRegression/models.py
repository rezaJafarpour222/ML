import numpy as np
from sklearn.linear_model import SGDRegressor

import myRegressor
from utils import splitter


def linearRegressor_from_sklearn(data, label, lr, epochs, splitPercent):

    (X_train, Y_train, X_test, Y_test) = splitter(
        data=data, label=label, splitPrecent=splitPercent
    )

    regressor = SGDRegressor(
        loss="squared_error",
        learning_rate="constant",
        penalty=None,
        eta0=lr,
        tol=False,
        max_iter=epochs,
        fit_intercept=False,
        random_state=0,
    )
    regressor.fit(X_train, Y_train)
    y_test_pred = regressor.predict(X_test)
    test_loss = np.mean((Y_test - y_test_pred) ** 2)

    y_train_pred = regressor.predict(X_train)
    train_loss = np.mean((Y_train - y_train_pred) ** 2)
    return (train_loss, test_loss, y_test_pred, Y_test)


def linearRegressor_my_model(data, label, lr, epochs, splitPercent):

    (X_train, Y_train, X_test, Y_test) = splitter(
        data=data, label=label, splitPrecent=splitPercent
    )

    model = myRegressor.LinearRegressor(data.shape[1])
    (_, _) = model.SGD(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        lr=lr,
        epochs=epochs,
    )
    y_predict = model.predict(X_train)
    train_loss = model.MSE(Y_train, y_predict)
    y_test_predict = model.predict(X_test)
    test_loss = model.MSE(Y_test, y_test_predict)
    return (train_loss, test_loss, y_test_predict, Y_test)
