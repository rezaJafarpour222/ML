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
