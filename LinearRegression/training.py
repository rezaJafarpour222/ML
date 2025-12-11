import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import Model
from utils import splitter


def linearRegressor_from_sklearn(data, label, lr, epochs, splitPercent):

    (X_train, Y_train, X_test, Y_test) = splitter(
        data=data, label=label, splitPrecent=splitPercent
    )

    regressor = SGDRegressor(
        loss="squared_error",
        learning_rate="constant",
        eta0=lr,
        max_iter=epochs,
        random_state=42,
    )
    regressor.fit(X_train, Y_train)
    y_test_pred = regressor.predict(X_test)
    test_loss = np.mean((Y_test - y_test_pred) ** 2)
    # test_loss = mean_squared_error(Y_test, y_test_pred)

    y_train_pred = regressor.predict(X_train)
    train_loss = np.mean((Y_train - y_train_pred) ** 2)
    # train_loss = mean_squared_error(Y_train, y_train_pred)
    return (train_loss, test_loss, y_test_pred, Y_test)


def linearRegressor_my_model(data, label, lr, epochs, splitPercent):

    (X_train, Y_train, X_test, Y_test) = splitter(
        data=data, label=label, splitPrecent=splitPercent
    )

    model = Model.LinearRegressor(data.shape[1])
    (_, _) = model.SGD(
        X_train_shuffled=X_train,
        Y_train_shuffled=Y_train,
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


def model_runner_for_each_data_percentage(data, label, lr, epoch, model_function):
    splits = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    model_train_losses = []
    model_test_losses = []
    for i in splits:
        (train_loss, test_loss, _, _) = model_function(data, label, lr, epoch, i)
        model_train_losses.append(train_loss)
        model_test_losses.append(test_loss)

    return (model_train_losses, model_test_losses, splits)


def model_runner_for_each_learning_rate(data, label, lr, epoch, model_function):
    lr = [
        0.001,
        0.002,
        0.003,
        0.004,
        0.005,
        0.010,
        0.015,
        0.020,
        0.025,
        0.030,
        0.035,
        0.040,
    ]
    model_train_losses = []
    model_test_losses = []
    for i in lr:
        (train_loss, test_loss, _, _) = model_function(data, label, i, epoch, 1.0)
        model_train_losses.append(train_loss)
        model_test_losses.append(test_loss)

    return (model_train_losses, model_test_losses, lr)
